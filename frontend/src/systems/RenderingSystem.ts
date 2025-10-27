/**
 * RenderingSystem - Manages the main render loop
 * Moved from main.ts renderLoop() function
 */

import * as THREE from 'three';
import type { OrbitControls } from 'three/addons';
import type { System, SystemContext } from '../types';
import { debug } from '../debug-logger';

export enum RenderMode {
    FUSION = 'FUSION',
    GAUSSIAN_ONLY = 'GAUSSIAN_ONLY',
    LOCAL_ONLY = 'LOCAL_ONLY',
    DEPTH_FUSION = 'DEPTH_FUSION',
}

export interface RenderingConfig {
    // Scenes
    localScene: THREE.Scene;
    fusionScene: THREE.Scene;
    debugScene: THREE.Scene;
    gaussianOnlyScene: THREE.Scene;
    depthFusionScene: THREE.Scene;

    // Cameras
    camera: THREE.PerspectiveCamera;
    orthoCamera: THREE.OrthographicCamera;
    debugCamera: THREE.OrthographicCamera;
    gaussianOnlyCamera: THREE.OrthographicCamera;
    depthFusionCamera: THREE.OrthographicCamera;

    // Render targets
    localRenderTarget: THREE.WebGLRenderTarget;

    // Controls
    controls: OrbitControls;
    clock: THREE.Clock;

    // UI elements
    renderFpsDiv: HTMLElement;
    depthDebugCheckbox: HTMLInputElement;

    // Callbacks
    onCameraUpdate?: () => void;
    onUpdate?: () => void;  // Called every frame
}

export class RenderingSystem implements System {
    name = 'rendering';

    private context!: SystemContext;
    private config!: RenderingConfig;
    private renderer!: THREE.WebGLRenderer;

    // State
    private currentRenderMode: RenderMode = RenderMode.FUSION;
    private frameCounter = 0;
    private currentTimeIndex = 0;
    private isPlaying = false;

    // FPS tracking
    private renderCnt = 0;
    private renderStart = 0;

    // Camera tracking
    private lastCameraUpdateTime = 0;
    private lastCameraPosition = new THREE.Vector3();
    private lastCameraTarget = new THREE.Vector3();
    private cameraUpdateInterval = 16.67; // ~60Hz

    async initialize(context: SystemContext): Promise<void> {
        this.context = context;
        this.renderer = context.renderingContext.renderer;
        debug.logMain('[RenderingSystem] Initialized');
    }

    /**
     * Configure the rendering system with scenes, cameras, etc.
     */
    configure(config: RenderingConfig): void {
        this.config = config;
        this.renderStart = performance.now();
        debug.logMain('[RenderingSystem] Configured');
    }

    /**
     * Set the current render mode
     */
    setRenderMode(mode: RenderMode): void {
        this.currentRenderMode = mode;
        debug.logMain(`[RenderingSystem] Render mode: ${mode}`);
    }

    /**
     * Update system state (called by Application.update())
     */
    update(deltaTime: number): void {
        // Update controls
        this.config.controls.update(deltaTime);

        // Update frame counter
        if (this.isPlaying) {
            this.frameCounter = (this.frameCounter + 1) % 300;
            this.currentTimeIndex = this.frameCounter / 299.0;
        }

        // FPS calculation
        const now = performance.now();
        const elapsed = now - this.renderStart;

        if (elapsed > 1000) {
            const duration = elapsed / 1000;
            const fps = this.renderCnt / duration;

            if (fps > 0 && fps <= 240 && isFinite(fps)) {
                this.config.renderFpsDiv.textContent = `Render FPS: ${fps.toFixed(2)}`;
            }

            this.renderCnt = 0;
            this.renderStart = now;
        }

        // Camera update check
        if (now - this.lastCameraUpdateTime > this.cameraUpdateInterval) {
            this.lastCameraPosition.copy(this.config.camera.position);
            this.lastCameraTarget.copy(this.config.controls.target);
            this.lastCameraUpdateTime = now;

            // Send camera frame to server
            if (this.config.onCameraUpdate) {
                this.config.onCameraUpdate();
            }

            // Emit camera update event
            this.context.eventBus.emit('camera:update', {
                position: this.config.camera.position.clone(),
                target: this.config.controls.target.clone(),
            });
        }

        // Call custom update callback
        if (this.config.onUpdate) {
            this.config.onUpdate();
        }
    }

    /**
     * Execute rendering (called by Application.render())
     */
    render(): void {
        this.renderCnt++;
        if (this.config.depthDebugCheckbox.checked) {
            // Depth debug mode
            this.renderer.setRenderTarget(this.config.localRenderTarget);
            this.renderer.render(this.config.localScene, this.config.camera);
            this.renderer.setRenderTarget(null);
            this.renderer.render(this.config.debugScene, this.config.debugCamera);
        } else {
            switch (this.currentRenderMode) {
                case RenderMode.FUSION:
                    this.renderer.setRenderTarget(this.config.localRenderTarget);
                    this.renderer.render(this.config.localScene, this.config.camera);
                    this.renderer.setRenderTarget(null);
                    this.renderer.render(this.config.fusionScene, this.config.orthoCamera);
                    break;

                case RenderMode.GAUSSIAN_ONLY:
                    this.renderer.render(
                        this.config.gaussianOnlyScene,
                        this.config.gaussianOnlyCamera
                    );
                    break;

                case RenderMode.LOCAL_ONLY:
                    this.renderer.render(this.config.localScene, this.config.camera);
                    break;

                case RenderMode.DEPTH_FUSION:
                    this.renderer.setRenderTarget(this.config.localRenderTarget);
                    this.renderer.render(this.config.localScene, this.config.camera);
                    this.renderer.setRenderTarget(null);
                    this.renderer.render(
                        this.config.depthFusionScene,
                        this.config.depthFusionCamera
                    );
                    break;
            }
        }
    }

    dispose(): void {
        debug.logMain('[RenderingSystem] Disposed');
    }
}
