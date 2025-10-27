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
    console.log("[TextureManager] updateFromVideoFrame called:", frame.frameId, frame.width, 'x', frame.height, {
      hasColorBitmap: !!frame.colorBitmap,
      hasDepthBitmap: !!frame.depthBitmap,
      hasDepthRaw: !!frame.depthRaw,
      hasColorData: !!frame.colorData,
      hasDepthData: !!frame.depthData
    });

    if (!this.colorTexture || !this.depthTexture) {
      console.error("[TextureManager] Textures not initialized");
      return;
    }

    // Update color texture
    // Priority 1: Use ImageBitmap (GPU optimized, from decode-worker)
    if (frame.colorBitmap) {
      console.log("[TextureManager] Updating color texture from ImageBitmap");
      this.colorTexture.image = frame.colorBitmap;
      this.colorTexture.needsUpdate = true;
    }
    // Priority 2: Use raw color data (fallback)
    else if (frame.colorData) {
      console.log("[TextureManager] Updating color texture from colorData, length:", frame.colorData.length);
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
    // Priority 1: Use raw Float16 data (JPEG mode)
    if (frame.depthRaw) {
      console.log("[TextureManager] Updating depth texture from depthRaw (Float16), size:", frame.depthRaw.length);

      // Verify texture type matches
      if (this.depthTexture.type !== THREE.HalfFloatType) {
        console.warn("[TextureManager] Depth texture type mismatch! Expected HalfFloatType, got", this.depthTexture.type);
      }

      // Copy data into existing array (don't replace reference)
      const currentData = this.depthTexture.image.data as Uint16Array;
      if (currentData.length !== frame.depthRaw.length) {
        console.warn("[TextureManager] Depth size mismatch:", currentData.length, "vs", frame.depthRaw.length);
        // Recreate texture with correct size
        this.depthTexture.image.data = new Uint16Array(frame.depthRaw);
      } else {
        currentData.set(frame.depthRaw);
      }

      this.depthTexture.needsUpdate = true;
    }
    // Priority 2: Use ImageBitmap (H264 mode with depth as ImageBitmap)
    else if (frame.depthBitmap) {
      console.log("[TextureManager] Updating depth texture from ImageBitmap");
      // Convert ImageBitmap to canvas for DataTexture
      const canvas = document.createElement("canvas");
      canvas.width = frame.depthBitmap.width;
      canvas.height = frame.depthBitmap.height;
      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.drawImage(frame.depthBitmap, 0, 0);
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        // Use red channel as depth
        const depthArray = new Uint8Array(canvas.width * canvas.height);
        for (let i = 0; i < depthArray.length; i++) {
          depthArray[i] = imageData.data[i * 4]; // Red channel
        }

        // Copy into existing array
        const currentData = this.depthTexture.image.data as Uint8Array;
        if (currentData.length !== depthArray.length) {
          this.depthTexture.image.data = depthArray;
        } else {
          currentData.set(depthArray);
        }

        this.depthTexture.needsUpdate = true;
      }
    }
    // Priority 3: Use converted depth data (fallback)
    else if (frame.depthData) {
      console.log("[TextureManager] Updating depth texture from depthData (fallback)");
      // Fallback: already converted Float32Array
      const depthArray = new Uint8Array(frame.depthData.buffer);

      // Copy into existing array
      const currentData = this.depthTexture.image.data as Uint8Array;
      if (currentData.length !== depthArray.length) {
        this.depthTexture.image.data = depthArray;
      } else {
        currentData.set(depthArray);
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
    console.log(`[TextureManager] setJpegMode(${enabled}) called, current mode: ${this.isJpegMode}`);

    if (this.isJpegMode === enabled) {
      console.log(`[TextureManager] Already in ${enabled ? 'JPEG' : 'H264'} mode, skipping`);
      return;
    }

    this.isJpegMode = enabled;
    console.log(`[TextureManager] Switching to ${enabled ? 'JPEG' : 'H264'} mode`);

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
        console.log(`[TextureManager] Created JPEG depth texture (Uint16Array, HalfFloatType): ${this.width}x${this.height}`);
      } else {
        this.depthTexture = new THREE.DataTexture(
          new Uint8Array(this.width * this.height),
          this.width,
          this.height,
          THREE.RedFormat,
          THREE.UnsignedByteType
        );
        console.log(`[TextureManager] Created H264 depth texture (Uint8Array, UnsignedByteType): ${this.width}x${this.height}`);
      }

      this.depthTexture.minFilter = THREE.NearestFilter;
      this.depthTexture.magFilter = THREE.NearestFilter;

      if (this.context) {
        this.context.renderingContext.registerTexture("wsDepth", this.depthTexture);
        console.log(`[TextureManager] Registered new depth texture with RenderingContext`);
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
