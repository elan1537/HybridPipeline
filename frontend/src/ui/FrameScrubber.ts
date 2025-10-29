/**
 * FrameScrubber - Timeline slider for frame navigation
 *
 * Provides:
 * - Timeline slider (0% ~ 100%)
 * - Frame number display (assuming 300 frames total)
 * - Drag to seek (pauses playback during drag)
 */

import { TimeController } from "../core/Application";

export class FrameScrubber {
  private container: HTMLDivElement;
  private slider: HTMLInputElement;
  private frameDisplay: HTMLSpanElement;
  private isDragging: boolean = false;

  private timeController: TimeController;
  private totalFrames: number = 300; // Default, can be configured

  constructor(timeController: TimeController, totalFrames: number = 300) {
    this.timeController = timeController;
    this.totalFrames = totalFrames;

    // Create container
    this.container = document.createElement("div");
    this.container.className = "frame-scrubber";

    // Create frame display
    this.frameDisplay = document.createElement("span");
    this.frameDisplay.className = "frame-display";
    this.frameDisplay.textContent = "Frame 0 / 299";

    // Create timeline slider
    this.slider = document.createElement("input");
    this.slider.type = "range";
    this.slider.min = "0";
    this.slider.max = "1000"; // High resolution for smooth scrubbing
    this.slider.value = "0";
    this.slider.className = "timeline-slider";

    // Event listeners
    this.slider.addEventListener("mousedown", () => this.onDragStart());
    this.slider.addEventListener("touchstart", () => this.onDragStart());
    this.slider.addEventListener("input", () => this.onSliderInput());
    this.slider.addEventListener("mouseup", () => this.onDragEnd());
    this.slider.addEventListener("touchend", () => this.onDragEnd());

    // Assemble UI
    const label = document.createElement("label");
    label.textContent = "Timeline";

    this.container.appendChild(label);
    this.container.appendChild(this.frameDisplay);
    this.container.appendChild(this.slider);
  }

  /**
   * Handle drag start - pause playback
   */
  private onDragStart(): void {
    this.isDragging = true;
    this.timeController.pause();
  }

  /**
   * Handle slider input - seek to position
   */
  private onSliderInput(): void {
    const sliderValue = parseInt(this.slider.value);
    const timeIndex = sliderValue / 1000.0; // [0, 1]

    this.timeController.seek(timeIndex);
    this.updateFrameDisplay(timeIndex);
  }

  /**
   * Handle drag end - optionally resume playback
   */
  private onDragEnd(): void {
    this.isDragging = false;
    // Note: User can manually press play button to resume
  }

  /**
   * Update frame display text
   */
  private updateFrameDisplay(timeIndex: number): void {
    const frameNumber = Math.floor(timeIndex * (this.totalFrames - 1));
    this.frameDisplay.textContent = `Frame ${frameNumber} / ${this.totalFrames - 1}`;
  }

  /**
   * Update UI from TimeController (call every frame)
   */
  update(): void {
    if (!this.isDragging) {
      const timeIndex = this.timeController.getCurrentTime();

      // Update slider position
      this.slider.value = Math.floor(timeIndex * 1000).toString();

      // Update frame display
      this.updateFrameDisplay(timeIndex);
    }
  }

  /**
   * Set total frames for display
   */
  setTotalFrames(totalFrames: number): void {
    this.totalFrames = totalFrames;
    this.update(); // Refresh display
  }

  /**
   * Get DOM element for mounting
   */
  getElement(): HTMLDivElement {
    return this.container;
  }

  /**
   * Show UI
   */
  show(): void {
    this.container.style.display = "block";
  }

  /**
   * Hide UI
   */
  hide(): void {
    this.container.style.display = "none";
  }

  /**
   * Dispose UI
   */
  dispose(): void {
    this.container.remove();
  }
}
