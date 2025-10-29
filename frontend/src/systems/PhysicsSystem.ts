/**
 * PhysicsSystem - Manages physics simulation and collision detection
 * CP3: Centralized physics management for Mesh-Gaussian collision
 */

import * as THREE from "three";
import { System, SystemContext } from "../types";
import { MeshGaussianCollision, CollisionResult } from "../physics/MeshGaussianCollision";
import { TextureManager } from "./TextureManager";
import { CameraController } from "./CameraController";

/**
 * Physics-enabled mesh object
 */
export class PhysicsMesh {
  mesh: THREE.Mesh;
  velocity: THREE.Vector3;
  acceleration: THREE.Vector3;
  mass: number;
  restitution: number; // Bounce coefficient [0, 1]
  friction: number; // Friction coefficient [0, 1]
  enabled: boolean;

  constructor(mesh: THREE.Mesh, options?: {
    velocity?: THREE.Vector3;
    acceleration?: THREE.Vector3;
    mass?: number;
    restitution?: number;
    friction?: number;
  }) {
    this.mesh = mesh;
    this.velocity = options?.velocity || new THREE.Vector3();
    this.acceleration = options?.acceleration || new THREE.Vector3(0, -9.8, 0); // Gravity
    this.mass = options?.mass || 1.0;
    this.restitution = options?.restitution || 0.5;
    this.friction = options?.friction || 0.3;
    this.enabled = true;
  }
}

export enum CollisionResponseType {
  Stop = "stop",
  Bounce = "bounce",
  Slide = "slide",
}

export class PhysicsSystem implements System {
  readonly name = "physics";

  private context: SystemContext | null = null;
  private collisionDetector: MeshGaussianCollision | null = null;
  private physicsMeshes: PhysicsMesh[] = [];

  // Physics configuration
  private gravity: THREE.Vector3 = new THREE.Vector3(0, -9.8, 0);
  private responseType: CollisionResponseType = CollisionResponseType.Stop;

  // Near/far planes (synced with camera)
  private near: number = 0.3;
  private far: number = 100.0;

  async initialize(context: SystemContext): Promise<void> {
    this.context = context;

    // Get required systems from context
    const textureManager = context.systems?.get("texture") as TextureManager;
    const cameraController = context.systems?.get("camera") as CameraController;

    if (!textureManager || !cameraController) {
      console.error("[PhysicsSystem] Required systems not available");
      return;
    }

    const camera = cameraController.getCamera();
    if (!camera) {
      console.error("[PhysicsSystem] Camera not available");
      return;
    }

    // Create collision detector
    this.collisionDetector = new MeshGaussianCollision(camera, textureManager);

    // Get camera near/far
    this.near = camera.near;
    this.far = camera.far;
    this.collisionDetector.updateClippingPlanes(this.near, this.far);

    console.log("[PhysicsSystem] Initialized with near:", this.near, "far:", this.far);
  }

  update(deltaTime: number): void {
    if (!this.collisionDetector) {
      return;
    }

    // Update physics for each mesh
    for (const physicsMesh of this.physicsMeshes) {
      if (!physicsMesh.enabled) {
        continue;
      }

      this.updatePhysicsMesh(physicsMesh, deltaTime);
    }
  }

  dispose(): void {
    this.physicsMeshes = [];
    this.collisionDetector = null;
  }

  // ========================================================================
  // Public API
  // ========================================================================

  /**
   * Add a mesh to physics simulation
   */
  addMesh(mesh: THREE.Mesh, options?: {
    velocity?: THREE.Vector3;
    acceleration?: THREE.Vector3;
    mass?: number;
    restitution?: number;
    friction?: number;
  }): PhysicsMesh {
    const physicsMesh = new PhysicsMesh(mesh, options);
    this.physicsMeshes.push(physicsMesh);
    console.log("[PhysicsSystem] Added physics mesh:", mesh.name || mesh.uuid);
    return physicsMesh;
  }

  /**
   * Remove a mesh from physics simulation
   */
  removeMesh(mesh: THREE.Mesh): boolean {
    const index = this.physicsMeshes.findIndex(pm => pm.mesh === mesh);
    if (index !== -1) {
      this.physicsMeshes.splice(index, 1);
      console.log("[PhysicsSystem] Removed physics mesh:", mesh.name || mesh.uuid);
      return true;
    }
    return false;
  }

  /**
   * Set global gravity
   */
  setGravity(gravity: THREE.Vector3): void {
    this.gravity.copy(gravity);

    // Update all meshes that use default gravity
    for (const physicsMesh of this.physicsMeshes) {
      physicsMesh.acceleration.copy(gravity);
    }
  }

  /**
   * Set collision response type
   */
  setResponseType(type: CollisionResponseType): void {
    this.responseType = type;
  }

  /**
   * Set collision epsilon (threshold)
   */
  setCollisionEpsilon(epsilon: number): void {
    if (this.collisionDetector) {
      this.collisionDetector.setEpsilon(epsilon);
    }
  }

  /**
   * Get all physics meshes
   */
  getPhysicsMeshes(): PhysicsMesh[] {
    return this.physicsMeshes;
  }

  // ========================================================================
  // Private Methods
  // ========================================================================

  private updatePhysicsMesh(physicsMesh: PhysicsMesh, deltaTime: number): void {
    // 1. Apply physics (velocity + acceleration)
    physicsMesh.velocity.add(
      physicsMesh.acceleration.clone().multiplyScalar(deltaTime)
    );

    // 2. Calculate proposed position
    const currentPos = physicsMesh.mesh.position.clone();
    const proposedPos = currentPos.clone().add(
      physicsMesh.velocity.clone().multiplyScalar(deltaTime)
    );

    // 3. Check collision
    const collision = this.collisionDetector!.checkMeshCollision(
      physicsMesh.mesh,
      proposedPos
    );

    if (collision.collision) {
      // 4. Handle collision
      this.handleCollision(physicsMesh, currentPos, proposedPos, collision);
    } else {
      // 5. No collision → apply movement
      physicsMesh.mesh.position.copy(proposedPos);
    }
  }

  private handleCollision(
    physicsMesh: PhysicsMesh,
    currentPos: THREE.Vector3,
    proposedPos: THREE.Vector3,
    collision: CollisionResult
  ): void {
    switch (this.responseType) {
      case CollisionResponseType.Stop:
        this.stopResponse(physicsMesh, currentPos);
        break;

      case CollisionResponseType.Bounce:
        this.bounceResponse(physicsMesh, currentPos, collision);
        break;

      case CollisionResponseType.Slide:
        this.slideResponse(physicsMesh, currentPos, proposedPos, collision);
        break;
    }
  }

  // ========================================================================
  // Collision Response Strategies
  // ========================================================================

  /**
   * Stop response: Mesh stops at collision point
   */
  private stopResponse(physicsMesh: PhysicsMesh, currentPos: THREE.Vector3): void {
    // Keep current position
    physicsMesh.mesh.position.copy(currentPos);

    // Stop movement
    physicsMesh.velocity.set(0, 0, 0);
  }

  /**
   * Bounce response: Mesh bounces off surface
   */
  private bounceResponse(
    physicsMesh: PhysicsMesh,
    currentPos: THREE.Vector3,
    collision: CollisionResult
  ): void {
    // Keep current position
    physicsMesh.mesh.position.copy(currentPos);

    // Get collision normal
    const normal = this.collisionDetector!.getCollisionNormal(
      collision.contactPoint || currentPos
    );

    if (normal) {
      // Reflect velocity
      physicsMesh.velocity.reflect(normal);

      // Apply restitution (energy loss)
      physicsMesh.velocity.multiplyScalar(physicsMesh.restitution);
    } else {
      // Fallback: stop
      physicsMesh.velocity.set(0, 0, 0);
    }
  }

  /**
   * Slide response: Mesh slides along surface
   */
  private slideResponse(
    physicsMesh: PhysicsMesh,
    currentPos: THREE.Vector3,
    proposedPos: THREE.Vector3,
    collision: CollisionResult
  ): void {
    // Get collision normal
    const normal = this.collisionDetector!.getCollisionNormal(
      collision.contactPoint || currentPos
    );

    if (!normal) {
      // Fallback: stop
      this.stopResponse(physicsMesh, currentPos);
      return;
    }

    // Calculate movement vector
    const movement = proposedPos.clone().sub(currentPos);

    // Project movement onto surface (remove normal component)
    const normalComponent = normal.clone().multiplyScalar(movement.dot(normal));
    const slideMovement = movement.clone().sub(normalComponent);

    // Apply friction
    slideMovement.multiplyScalar(1.0 - physicsMesh.friction);

    // Update position
    physicsMesh.mesh.position.copy(currentPos.add(slideMovement));

    // Update velocity (project onto surface)
    const velocityNormalComponent = normal.clone().multiplyScalar(
      physicsMesh.velocity.dot(normal)
    );
    physicsMesh.velocity.sub(velocityNormalComponent);
  }
}
