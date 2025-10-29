/**
 * MeshGaussianCollision - Depth-based collision detection
 *
 * Detects collision between Three.js meshes and Gaussian Splatting scene
 * using depth map comparison.
 */

import * as THREE from "three";
import { TextureManager } from "../systems/TextureManager";

export interface CollisionResult {
  collision: boolean;
  gaussianDepth?: number;
  meshDepth?: number;
  penetrationDepth?: number;
  contactPoint?: THREE.Vector3;
}

export class MeshGaussianCollision {
  private camera: THREE.PerspectiveCamera;
  private textureManager: TextureManager;
  private near: number = 0.3;
  private far: number = 100.0;

  // Collision threshold (meters)
  private epsilon: number = 0.01; // 1cm tolerance

  constructor(camera: THREE.PerspectiveCamera, textureManager: TextureManager) {
    this.camera = camera;
    this.textureManager = textureManager;
  }

  /**
   * Update near/far clipping planes
   * Should be called when camera parameters change
   */
  updateClippingPlanes(near: number, far: number): void {
    this.near = near;
    this.far = far;
  }

  /**
   * Set collision threshold
   */
  setEpsilon(epsilon: number): void {
    this.epsilon = epsilon;
  }

  /**
   * Check collision for a single point
   *
   * @param worldPos - World space position to test
   * @returns Collision result
   */
  checkCollision(worldPos: THREE.Vector3): CollisionResult {
    // 1. Project world position to screen space
    const screenUV = this.worldToScreen(worldPos);

    // 2. Check if position is within screen bounds
    if (screenUV.x < 0 || screenUV.x > 1 || screenUV.y < 0 || screenUV.y > 1) {
      return { collision: false };
    }

    // 3. Get resolution
    const { width, height } = this.textureManager.getResolution();

    // 4. Sample Gaussian depth at screen position
    const x = Math.floor(screenUV.x * width);
    const y = Math.floor(screenUV.y * height);
    const gaussianDepth = this.textureManager.sampleLinearDepth(x, y, this.near, this.far);

    if (gaussianDepth === null) {
      return { collision: false };
    }

    // 5. Calculate mesh depth (camera distance)
    const meshDepth = this.camera.position.distanceTo(worldPos);

    // 6. Check collision
    // Mesh is behind Gaussian surface → collision
    if (meshDepth >= gaussianDepth - this.epsilon) {
      return {
        collision: true,
        gaussianDepth,
        meshDepth,
        penetrationDepth: meshDepth - gaussianDepth,
        contactPoint: worldPos.clone(),
      };
    }

    return { collision: false };
  }

  /**
   * Check collision for mesh using bounding box
   *
   * Samples multiple points on the bounding box to detect collision
   *
   * @param mesh - Three.js mesh to test
   * @param proposedPosition - Proposed world position (before applying)
   * @returns Collision result (collision is true if ANY point collides)
   */
  checkMeshCollision(
    mesh: THREE.Mesh,
    proposedPosition: THREE.Vector3
  ): CollisionResult {
    // Get bounding box in local space
    if (!mesh.geometry.boundingBox) {
      mesh.geometry.computeBoundingBox();
    }

    const bbox = mesh.geometry.boundingBox!;

    // Sample 8 corners of bounding box
    const corners = [
      new THREE.Vector3(bbox.min.x, bbox.min.y, bbox.min.z),
      new THREE.Vector3(bbox.max.x, bbox.min.y, bbox.min.z),
      new THREE.Vector3(bbox.min.x, bbox.max.y, bbox.min.z),
      new THREE.Vector3(bbox.max.x, bbox.max.y, bbox.min.z),
      new THREE.Vector3(bbox.min.x, bbox.min.y, bbox.max.z),
      new THREE.Vector3(bbox.max.x, bbox.min.y, bbox.max.z),
      new THREE.Vector3(bbox.min.x, bbox.max.y, bbox.max.z),
      new THREE.Vector3(bbox.max.x, bbox.max.y, bbox.max.z),
    ];

    // Transform corners to world space at proposed position
    const worldCorners: THREE.Vector3[] = [];
    const tempMatrix = new THREE.Matrix4();
    tempMatrix.compose(
      proposedPosition,
      mesh.quaternion,
      mesh.scale
    );

    for (const corner of corners) {
      const worldCorner = corner.clone().applyMatrix4(tempMatrix);
      worldCorners.push(worldCorner);
    }

    // Check collision for each corner
    let maxPenetration = -Infinity;
    let worstCollision: CollisionResult | null = null;

    for (const worldCorner of worldCorners) {
      const result = this.checkCollision(worldCorner);

      if (result.collision && result.penetrationDepth !== undefined) {
        if (result.penetrationDepth > maxPenetration) {
          maxPenetration = result.penetrationDepth;
          worstCollision = result;
        }
      }
    }

    if (worstCollision) {
      return worstCollision;
    }

    return { collision: false };
  }

  /**
   * Check collision for mesh using center + radius (sphere approximation)
   *
   * Faster than bounding box, useful for circular objects
   *
   * @param center - Center position in world space
   * @param radius - Collision radius
   * @returns Collision result
   */
  checkSphereCollision(center: THREE.Vector3, radius: number): CollisionResult {
    // Sample center point
    const centerResult = this.checkCollision(center);

    if (!centerResult.collision) {
      return { collision: false };
    }

    // If center collides, check if sphere surface also collides
    // by adding radius to penetration depth
    const penetrationDepth = (centerResult.penetrationDepth || 0) + radius;

    return {
      collision: true,
      gaussianDepth: centerResult.gaussianDepth,
      meshDepth: centerResult.meshDepth,
      penetrationDepth,
      contactPoint: center.clone(),
    };
  }

  /**
   * Get collision normal (direction to push mesh away from surface)
   *
   * @param worldPos - Collision point
   * @returns Normal vector pointing away from Gaussian surface
   */
  getCollisionNormal(worldPos: THREE.Vector3): THREE.Vector3 | null {
    const screenUV = this.worldToScreen(worldPos);

    if (screenUV.x < 0 || screenUV.x > 1 || screenUV.y < 0 || screenUV.y > 1) {
      return null;
    }

    const { width, height } = this.textureManager.getResolution();

    // Sample depth gradient (finite differences)
    const u = screenUV.x;
    const v = screenUV.y;
    const delta = 1.0 / width; // 1 pixel

    const d_center = this.textureManager.sampleLinearDepthBilinear(u, v, this.near, this.far);
    const d_right = this.textureManager.sampleLinearDepthBilinear(u + delta, v, this.near, this.far);
    const d_up = this.textureManager.sampleLinearDepthBilinear(u, v + delta, this.near, this.far);

    if (d_center === null || d_right === null || d_up === null) {
      // Fallback: push towards camera
      return this.camera.position.clone().sub(worldPos).normalize();
    }

    // Compute gradient in screen space
    const grad_x = d_right - d_center;
    const grad_y = d_up - d_center;

    // Convert screen-space gradient to world-space normal
    // For simplicity, use camera direction as base
    const cameraDir = new THREE.Vector3()
      .subVectors(worldPos, this.camera.position)
      .normalize();

    // If depth increases in gradient direction, push away from camera
    // This is a simplified approximation
    if (grad_x > 0 || grad_y > 0) {
      return cameraDir.negate();
    } else {
      return cameraDir;
    }
  }

  // ========================================================================
  // Private Methods
  // ========================================================================

  /**
   * Project world position to screen UV coordinates
   *
   * @param worldPos - World space position
   * @returns Screen UV [0, 1] x [0, 1]
   */
  private worldToScreen(worldPos: THREE.Vector3): THREE.Vector2 {
    const projected = worldPos.clone().project(this.camera);

    // NDC [-1, 1] → UV [0, 1]
    return new THREE.Vector2(
      (projected.x + 1) / 2,
      (1 - projected.y) / 2 // Y axis flip
    );
  }
}
