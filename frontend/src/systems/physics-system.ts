/**
 * PhysicsSystem - Rapier physics engine integration
 * Follows the System interface pattern from the existing codebase
 */

import RAPIER from "@dimforge/rapier3d-compat";
import * as THREE from "three";
import {
  System,
  SystemContext,
  PhysicsConfig,
  PhysicsBodyOptions,
  PhysicsBodySync,
  PhysicsMode,
} from "../types";

export class PhysicsSystem implements System {
  readonly name = "physics";

  private context: SystemContext | null = null;
  private world: RAPIER.World | null = null;
  private bodies: Map<string, PhysicsBodySync> = new Map();
  private scene: THREE.Scene | null = null;

  // Debug visualization
  private debugMesh: THREE.LineSegments | null = null;
  private debugEnabled: boolean = false;

  // Physics configuration
  private config: PhysicsConfig = {
    gravity: { x: 0, y: -9.81, z: 0 },
    timestep: 1 / 60,
    substeps: 1,
  };

  // Fixed timestep accumulator
  private accumulator: number = 0;

  async initialize(context: SystemContext): Promise<void> {
    this.context = context;
    this.scene = context.renderingContext?.scene || null;

    // Get physics mode from state
    const physicsMode = context.state.get("physics:mode") as PhysicsMode;
    if (physicsMode === PhysicsMode.DISABLED) {
      console.log("[PhysicsSystem] Physics disabled, skipping initialization");
      return;
    }

    console.log("[PhysicsSystem] Initializing Rapier WASM...");
    await RAPIER.init();
    console.log("[PhysicsSystem] Rapier WASM initialized");

    // Create physics world
    const gravity = new RAPIER.Vector3(
      this.config.gravity.x,
      this.config.gravity.y,
      this.config.gravity.z
    );
    this.world = new RAPIER.World(gravity);

    // Enable debug visualization for DEBUG and DEBUG_ONLY modes
    if (
      physicsMode === PhysicsMode.DEBUG ||
      physicsMode === PhysicsMode.DEBUG_ONLY
    ) {
      this.enableDebugRenderer();
    }

    console.log("[PhysicsSystem] Initialized with mode:", physicsMode);
  }

  update(deltaTime: number): void {
    if (!this.world) {
      return;
    }

    // Fixed timestep accumulator pattern
    this.accumulator += deltaTime;

    while (this.accumulator >= this.config.timestep) {
      this.world.step();
      this.accumulator -= this.config.timestep;
    }

    // Sync Rapier bodies → Three.js objects
    this.syncBodiesToThreeJS();

    // Update debug visualization
    if (this.debugEnabled && this.debugMesh) {
      this.updateDebugRenderer();
    }
  }

  dispose(): void {
    // Remove debug mesh
    if (this.debugMesh && this.scene) {
      this.scene.remove(this.debugMesh);
      this.debugMesh.geometry.dispose();
      (this.debugMesh.material as THREE.Material).dispose();
      this.debugMesh = null;
    }

    // Clear body mappings
    this.bodies.clear();

    // Free Rapier world
    if (this.world) {
      this.world.free();
      this.world = null;
    }

    console.log("[PhysicsSystem] Disposed");
  }

  // ========================================================================
  // Public API
  // ========================================================================

  /**
   * Add physics body to an Object3D
   */
  addBody(object3D: THREE.Object3D, options: PhysicsBodyOptions): number | null {
    if (!this.world) {
      console.warn("[PhysicsSystem] Cannot add body: world not initialized");
      return null;
    }

    // Remove existing body if present
    if (this.bodies.has(object3D.uuid)) {
      this.removeBody(object3D);
    }

    // Get world position/rotation from Object3D
    object3D.updateMatrixWorld(true);
    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    const scale = new THREE.Vector3();
    object3D.matrixWorld.decompose(position, quaternion, scale);

    // Create rigid body description
    let rigidBodyDesc: RAPIER.RigidBodyDesc;
    switch (options.type) {
      case "dynamic":
        rigidBodyDesc = RAPIER.RigidBodyDesc.dynamic();
        break;
      case "kinematic":
        rigidBodyDesc = RAPIER.RigidBodyDesc.kinematicPositionBased();
        break;
      case "static":
      default:
        rigidBodyDesc = RAPIER.RigidBodyDesc.fixed();
        break;
    }

    rigidBodyDesc.setTranslation(position.x, position.y, position.z);
    rigidBodyDesc.setRotation({
      x: quaternion.x,
      y: quaternion.y,
      z: quaternion.z,
      w: quaternion.w,
    });

    // Create rigid body
    const rigidBody = this.world.createRigidBody(rigidBodyDesc);

    // Create collider
    const colliderDesc = this.createColliderDesc(object3D, options, scale);
    if (options.restitution !== undefined) {
      colliderDesc.setRestitution(options.restitution);
    }
    if (options.friction !== undefined) {
      colliderDesc.setFriction(options.friction);
    }

    const collider = this.world.createCollider(colliderDesc, rigidBody);

    // Store mapping
    const sync: PhysicsBodySync = {
      object3D,
      rigidBodyHandle: rigidBody.handle,
      colliderHandle: collider.handle,
      options,
    };

    this.bodies.set(object3D.uuid, sync);

    console.log(
      `[PhysicsSystem] Added ${options.type} body for "${object3D.name || object3D.uuid}"`
    );

    return rigidBody.handle;
  }

  /**
   * Remove physics body from an Object3D
   */
  removeBody(object3D: THREE.Object3D): void {
    const sync = this.bodies.get(object3D.uuid);
    if (!sync || !this.world) return;

    const rigidBody = this.world.getRigidBody(sync.rigidBodyHandle);
    if (rigidBody) {
      this.world.removeRigidBody(rigidBody);
    }

    this.bodies.delete(object3D.uuid);
    console.log(
      `[PhysicsSystem] Removed body for "${object3D.name || object3D.uuid}"`
    );
  }

  /**
   * Apply force to a body
   */
  applyForce(object3D: THREE.Object3D, force: THREE.Vector3): void {
    const sync = this.bodies.get(object3D.uuid);
    if (!sync || !this.world) return;

    const rigidBody = this.world.getRigidBody(sync.rigidBodyHandle);
    if (rigidBody && rigidBody.isDynamic()) {
      rigidBody.addForce({ x: force.x, y: force.y, z: force.z }, true);
    }
  }

  /**
   * Apply impulse to a body
   */
  applyImpulse(object3D: THREE.Object3D, impulse: THREE.Vector3): void {
    const sync = this.bodies.get(object3D.uuid);
    if (!sync || !this.world) return;

    const rigidBody = this.world.getRigidBody(sync.rigidBodyHandle);
    if (rigidBody && rigidBody.isDynamic()) {
      rigidBody.applyImpulse({ x: impulse.x, y: impulse.y, z: impulse.z }, true);
    }
  }

  /**
   * Set gravity
   */
  setGravity(x: number, y: number, z: number): void {
    if (this.world) {
      this.world.gravity = new RAPIER.Vector3(x, y, z);
      this.config.gravity = { x, y, z };
    }
  }

  /**
   * Get physics world (for advanced use)
   */
  getWorld(): RAPIER.World | null {
    return this.world;
  }

  /**
   * Check if physics is enabled
   */
  isEnabled(): boolean {
    return this.world !== null;
  }

  /**
   * Enable/disable debug visualization
   */
  setDebugEnabled(enabled: boolean): void {
    this.debugEnabled = enabled;
    if (this.debugMesh) {
      this.debugMesh.visible = enabled;
    } else if (enabled) {
      this.enableDebugRenderer();
    }
  }

  // ========================================================================
  // Private Methods
  // ========================================================================

  /**
   * Create collider description based on shape type
   */
  private createColliderDesc(
    object3D: THREE.Object3D,
    options: PhysicsBodyOptions,
    scale: THREE.Vector3
  ): RAPIER.ColliderDesc {
    const args = options.colliderArgs || [];

    switch (options.colliderShape) {
      case "box": {
        // Auto-calculate from bounding box if no args provided
        if (args.length === 0 && object3D instanceof THREE.Mesh) {
          const box = new THREE.Box3().setFromObject(object3D);
          const size = box.getSize(new THREE.Vector3());
          return RAPIER.ColliderDesc.cuboid(
            (size.x * scale.x) / 2,
            (size.y * scale.y) / 2,
            (size.z * scale.z) / 2
          );
        }
        return RAPIER.ColliderDesc.cuboid(
          (args[0] || 0.5) * scale.x,
          (args[1] || 0.5) * scale.y,
          (args[2] || 0.5) * scale.z
        );
      }

      case "sphere": {
        // Auto-calculate from bounding sphere if no args provided
        if (args.length === 0 && object3D instanceof THREE.Mesh) {
          const sphere = new THREE.Sphere();
          object3D.geometry.computeBoundingSphere();
          if (object3D.geometry.boundingSphere) {
            sphere.copy(object3D.geometry.boundingSphere);
            return RAPIER.ColliderDesc.ball(
              sphere.radius * Math.max(scale.x, scale.y, scale.z)
            );
          }
        }
        return RAPIER.ColliderDesc.ball(
          (args[0] || 0.5) * Math.max(scale.x, scale.y, scale.z)
        );
      }

      case "capsule": {
        const halfHeight = (args[0] || 0.5) * scale.y;
        const radius = (args[1] || 0.25) * Math.max(scale.x, scale.z);
        return RAPIER.ColliderDesc.capsule(halfHeight, radius);
      }

      case "mesh": {
        if (object3D instanceof THREE.Mesh) {
          const geometry = object3D.geometry;
          const positions = geometry.getAttribute("position");
          const indices = geometry.getIndex();

          if (positions && indices) {
            const vertices = new Float32Array(positions.array);
            const indicesArray = new Uint32Array(indices.array);
            return RAPIER.ColliderDesc.trimesh(vertices, indicesArray);
          }
        }
        // Fallback to box
        return RAPIER.ColliderDesc.cuboid(0.5, 0.5, 0.5);
      }

      case "convex": {
        if (object3D instanceof THREE.Mesh) {
          const positions = object3D.geometry.getAttribute("position");
          if (positions) {
            const vertices = new Float32Array(positions.array);
            return (
              RAPIER.ColliderDesc.convexHull(vertices) ||
              RAPIER.ColliderDesc.cuboid(0.5, 0.5, 0.5)
            );
          }
        }
        return RAPIER.ColliderDesc.cuboid(0.5, 0.5, 0.5);
      }

      default:
        return RAPIER.ColliderDesc.cuboid(0.5, 0.5, 0.5);
    }
  }

  /**
   * Sync all Rapier bodies to Three.js objects
   */
  private syncBodiesToThreeJS(): void {
    if (!this.world) return;

    for (const [_uuid, sync] of this.bodies) {
      const rigidBody = this.world.getRigidBody(sync.rigidBodyHandle);
      if (!rigidBody) continue;

      // Skip kinematic bodies (controlled by Three.js)
      if (sync.options.type === "kinematic") {
        // Update Rapier from Three.js position
        const pos = sync.object3D.position;
        const quat = sync.object3D.quaternion;
        rigidBody.setNextKinematicTranslation({ x: pos.x, y: pos.y, z: pos.z });
        rigidBody.setNextKinematicRotation({
          x: quat.x,
          y: quat.y,
          z: quat.z,
          w: quat.w,
        });
      } else if (sync.options.type === "dynamic") {
        // Update Three.js from Rapier position
        const translation = rigidBody.translation();
        const rotation = rigidBody.rotation();

        sync.object3D.position.set(translation.x, translation.y, translation.z);
        sync.object3D.quaternion.set(
          rotation.x,
          rotation.y,
          rotation.z,
          rotation.w
        );
      }
      // Static bodies don't need syncing
    }
  }

  /**
   * Enable debug renderer
   */
  private enableDebugRenderer(): void {
    if (!this.scene) return;

    const geometry = new THREE.BufferGeometry();
    const material = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.8,
    });

    this.debugMesh = new THREE.LineSegments(geometry, material);
    this.debugMesh.frustumCulled = false;
    this.debugMesh.name = "PhysicsDebugMesh";

    this.scene.add(this.debugMesh);
    this.debugEnabled = true;

    console.log("[PhysicsSystem] Debug renderer enabled");
  }

  /**
   * Update debug renderer
   */
  private updateDebugRenderer(): void {
    if (!this.world || !this.debugMesh) return;

    const buffers = this.world.debugRender();

    this.debugMesh.geometry.setAttribute(
      "position",
      new THREE.BufferAttribute(buffers.vertices, 3)
    );

    this.debugMesh.geometry.setAttribute(
      "color",
      new THREE.BufferAttribute(buffers.colors, 4)
    );

    this.debugMesh.geometry.computeBoundingSphere();
  }
}
