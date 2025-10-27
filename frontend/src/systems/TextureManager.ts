/**
 * TextureManager - Manages WebSocket textures (color and depth)
 * CP3: Centralized texture management for wsColorTexture and wsDepthTexture
 */

import * as THREE from "three";
import { System, SystemContext, VideoFrame, FormatType } from "../types";

export class TextureManager implements System {
  readonly name = "texture";

  private context: SystemContext | null = null;
  private colorTexture: THREE.Texture | null = null;
  private depthTexture: THREE.DataTexture | null = null;

  private width: number = 1280;
  private height: number = 1280;
  private isJpegMode: boolean = false;

  async initialize(context: SystemContext): Promise<void> {
    this.context = context;

    // Get initial size from state
    this.width = context.state.getOrDefault("rendering:width", 1280);
    this.height = context.state.getOrDefault("rendering:height", 720);

    // Create initial textures
    this.createTextures();

    // Subscribe to resolution changes
    context.state.subscribe("rendering:width", (width: number) => {
      this.resize(width, this.height);
    });

    context.state.subscribe("rendering:height", (height: number) => {
      this.resize(this.width, height);
    });

    console.log("[TextureManager] Initialized with size", this.width, "x", this.height);
  }

  update(deltaTime: number): void {
    // Check if textures need update from state
    const colorNeedsUpdate = this.context?.state.get("texture:colorNeedsUpdate");
    const depthNeedsUpdate = this.context?.state.get("texture:depthNeedsUpdate");

    if (colorNeedsUpdate && this.colorTexture) {
      this.colorTexture.needsUpdate = true;
      this.context?.state.set("texture:colorNeedsUpdate", false);
    }

    if (depthNeedsUpdate && this.depthTexture) {
      this.depthTexture.needsUpdate = true;
      this.context?.state.set("texture:depthNeedsUpdate", false);
    }
  }

  dispose(): void {
    this.disposeTextures();
  }

  // ========================================================================
  // Texture Creation and Management
  // ========================================================================

  private createTextures(): void {
    // Create color texture
    this.colorTexture = new THREE.Texture();
    this.colorTexture.minFilter = THREE.LinearFilter;
    this.colorTexture.magFilter = THREE.LinearFilter;
    this.colorTexture.colorSpace = THREE.SRGBColorSpace;

    // Create depth texture (format depends on mode)
    if (this.isJpegMode) {
      // JPEG mode: Float16 data in Uint16Array format
      this.depthTexture = new THREE.DataTexture(
        new Uint16Array(this.width * this.height),
        this.width,
        this.height,
        THREE.RedFormat,
        THREE.HalfFloatType
      );
    } else {
      // H264 mode: Uint8Array grayscale data
      this.depthTexture = new THREE.DataTexture(
        new Uint8Array(this.width * this.height),
        this.width,
        this.height,
        THREE.RedFormat,
        THREE.UnsignedByteType
      );
    }

    this.depthTexture.minFilter = THREE.NearestFilter;
    this.depthTexture.magFilter = THREE.NearestFilter;

    // Register with rendering context
    if (this.context) {
      this.context.renderingContext.registerTexture("wsColor", this.colorTexture);
      this.context.renderingContext.registerTexture("wsDepth", this.depthTexture);
    }
  }

  private disposeTextures(): void {
    if (this.colorTexture) {
      this.colorTexture.dispose();
      this.colorTexture = null;
    }

    if (this.depthTexture) {
      this.depthTexture.dispose();
      this.depthTexture = null;
    }
  }

  // ========================================================================
  // Public API
  // ========================================================================

  /**
   * Update textures from video frame
   */
  updateFromVideoFrame(frame: VideoFrame): void {
    if (!this.colorTexture || !this.depthTexture) {
      console.error("[TextureManager] Textures not initialized");
      return;
    }

    // Update color texture
    if (frame.colorData) {
      const colorImage = new ImageData(
        new Uint8ClampedArray(frame.colorData),
        frame.width,
        frame.height
      );

      // Create canvas to convert ImageData to Image
      const canvas = document.createElement("canvas");
      canvas.width = frame.width;
      canvas.height = frame.height;
      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.putImageData(colorImage, 0, 0);
        this.colorTexture.image = canvas;
        this.colorTexture.needsUpdate = true;
      }
    }

    // Update depth texture
    if (frame.depthData) {
      if (this.isJpegMode) {
        // JPEG mode: Float16 data
        const depthArray = new Uint16Array(frame.depthData.buffer);
        this.depthTexture.image.data = depthArray;
      } else {
        // H264 mode: Uint8 data
        const depthArray = new Uint8Array(frame.depthData.buffer);
        this.depthTexture.image.data = depthArray;
      }
      this.depthTexture.needsUpdate = true;
    }

    // Update state
    if (this.context) {
      this.context.state.set("texture:lastFrameId", frame.frameId);
    }
  }

  /**
   * Resize textures
   */
  resize(width: number, height: number): void {
    if (this.width === width && this.height === height) {
      return;
    }

    console.log("[TextureManager] Resizing from", this.width, "x", this.height, "to", width, "x", height);

    this.width = width;
    this.height = height;

    // Recreate textures with new size
    this.disposeTextures();
    this.createTextures();
  }

  /**
   * Set JPEG mode
   */
  setJpegMode(enabled: boolean): void {
    if (this.isJpegMode === enabled) {
      return;
    }

    this.isJpegMode = enabled;

    // Recreate depth texture with correct format
    if (this.depthTexture) {
      this.depthTexture.dispose();

      if (this.isJpegMode) {
        this.depthTexture = new THREE.DataTexture(
          new Uint16Array(this.width * this.height),
          this.width,
          this.height,
          THREE.RedFormat,
          THREE.HalfFloatType
        );
      } else {
        this.depthTexture = new THREE.DataTexture(
          new Uint8Array(this.width * this.height),
          this.width,
          this.height,
          THREE.RedFormat,
          THREE.UnsignedByteType
        );
      }

      this.depthTexture.minFilter = THREE.NearestFilter;
      this.depthTexture.magFilter = THREE.NearestFilter;

      if (this.context) {
        this.context.renderingContext.registerTexture("wsDepth", this.depthTexture);
      }
    }
  }

  // ========================================================================
  // Getters
  // ========================================================================

  getColorTexture(): THREE.Texture | null {
    return this.colorTexture;
  }

  getDepthTexture(): THREE.DataTexture | null {
    return this.depthTexture;
  }

  getResolution(): { width: number; height: number } {
    return { width: this.width, height: this.height };
  }

  isJPEGMode(): boolean {
    return this.isJpegMode;
  }
}
