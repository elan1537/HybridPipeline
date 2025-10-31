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

    // Initial shader material update will be called after materials are registered
    // This is deferred to allow main.ts to register materials first
    console.log("[TextureManager] Initialized with size", this.width, "x", this.height);
    console.log("[TextureManager] Shader materials will be updated after registration");

    // Subscribe to resolution changes
    context.state.subscribe("rendering:width", (width: number) => {
      this.resize(width, this.height);
    });

    context.state.subscribe("rendering:height", (height: number) => {
      this.resize(this.width, height);
    });
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

        // Update all shader materials that use this depth texture
        this.updateShaderMaterials();
      }
    }
  }

  /**
   * Update shader materials to use the current textures (both color and depth)
   * This is called when textures are created/recreated (e.g., initialization, mode switch, resize)
   */
  private updateShaderMaterials(): void {
    if (!this.context || !this.colorTexture || !this.depthTexture) {
      console.warn('[TextureManager] Cannot update shader materials: textures not ready');
      return;
    }

    const renderingContext = this.context.renderingContext;

    // Material configuration: which uniforms each material needs
    const materialConfig = [
      { name: 'fusion', uniforms: ['wsColorSampler', 'wsDepthSampler'] },
      { name: 'debug', uniforms: ['wsColorSampler', 'wsDepthSampler'] },
      { name: 'depthFusion', uniforms: ['wsColorSampler', 'wsDepthSampler'] },
      { name: 'gaussianOnly', uniforms: ['wsColorSampler'] },
    ];

    materialConfig.forEach(config => {
      const material = renderingContext.getMaterial(config.name);
      if (material && 'uniforms' in material) {
        const shaderMaterial = material as THREE.ShaderMaterial;

        config.uniforms.forEach(uniformName => {
          if (shaderMaterial.uniforms[uniformName]) {
            if (uniformName === 'wsColorSampler') {
              shaderMaterial.uniforms[uniformName].value = this.colorTexture;
              console.log(`[TextureManager] Updated ${config.name}.${uniformName}`);
            } else if (uniformName === 'wsDepthSampler') {
              shaderMaterial.uniforms[uniformName].value = this.depthTexture;
              console.log(`[TextureManager] Updated ${config.name}.${uniformName}`);
            }
          }
        });
      }
    });

    console.log(`[TextureManager] All shader materials updated with current textures`);
  }

  /**
   * Called by main.ts after shader materials are registered with RenderingContext
   * This performs the initial texture assignment
   */
  initializeShaderMaterials(): void {
    console.log('[TextureManager] initializeShaderMaterials() called');
    this.updateShaderMaterials();
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

  // ========================================================================
  // Depth Sampling (for Collision Detection)
  // ========================================================================

  /**
   * Sample linear depth value at pixel coordinates
   *
   * Backend encodes depth as: d_norm = (log(depth/near) / log(far/near)) * 0.80823
   * Inverse: depth = near * exp((d_norm / 0.80823) * log(far/near))
   *
   * @param x - Pixel x coordinate [0, width)
   * @param y - Pixel y coordinate [0, height)
   * @param near - Near clipping plane (meters)
   * @param far - Far clipping plane (meters)
   * @returns Linear depth in meters, or null if out of bounds or no data
   */
  sampleLinearDepth(x: number, y: number, near: number, far: number): number | null {
    if (!this.depthTexture || x < 0 || x >= this.width || y < 0 || y >= this.height) {
      return null;
    }

    const pixelIndex = Math.floor(y) * this.width + Math.floor(x);
    const data = this.depthTexture.image.data as Uint16Array | Uint8Array;

    if (!data || pixelIndex >= data.length) {
      return null;
    }

    let normalized: number;

    if (this.isJpegMode) {
      // JPEG mode: Uint16Array (Float16 encoded)
      const uint16Data = data as Uint16Array;
      const uint16Value = uint16Data[pixelIndex];

      // Convert Float16 to Float32 (inline to avoid dependency)
      // Simple approximation for normalized [0, 1] range
      // For full precision, use proper IEEE 754 Float16 decoder
      normalized = uint16Value / 65535.0; // Simple linear approximation
    } else {
      // H264 mode: Uint8Array
      const uint8Data = data as Uint8Array;
      normalized = uint8Data[pixelIndex] / 255.0;
    }

    // Convert log-normalized depth to linear depth
    const scale = 0.80823; // Backend scale factor
    if (normalized <= 0) return near;
    if (normalized >= scale) return far;

    const logRatio = (normalized / scale) * Math.log(far / near);
    const linearDepth = near * Math.exp(logRatio);

    return linearDepth;
  }

  /**
   * Sample depth with bilinear interpolation
   *
   * @param u - Normalized texture coordinate [0, 1]
   * @param v - Normalized texture coordinate [0, 1]
   * @param near - Near clipping plane
   * @param far - Far clipping plane
   * @returns Interpolated linear depth, or null if invalid
   */
  sampleLinearDepthBilinear(u: number, v: number, near: number, far: number): number | null {
    const x = u * this.width - 0.5;
    const y = v * this.height - 0.5;

    const x0 = Math.floor(x);
    const y0 = Math.floor(y);
    const x1 = x0 + 1;
    const y1 = y0 + 1;
    const fx = x - x0;
    const fy = y - y0;

    const d00 = this.sampleLinearDepth(x0, y0, near, far);
    const d10 = this.sampleLinearDepth(x1, y0, near, far);
    const d01 = this.sampleLinearDepth(x0, y1, near, far);
    const d11 = this.sampleLinearDepth(x1, y1, near, far);

    if (d00 === null || d10 === null || d01 === null || d11 === null) {
      return null;
    }

    const d0 = d00 * (1 - fx) + d10 * fx;
    const d1 = d01 * (1 - fx) + d11 * fx;
    return d0 * (1 - fy) + d1 * fy;
  }
}
