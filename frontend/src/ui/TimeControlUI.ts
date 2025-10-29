/**
 * TimeControlUI - UI component for controlling time playback
 *
 * Provides:
 * - Play/Pause button
 * - Playback speed slider (0.1x ~ 5.0x)
 * - Current time display
 */

import { TimeController } from "../core/Application";

export class TimeControlUI {
  private container: HTMLDivElement;
  private playButton: HTMLButtonElement;
  private speedSlider: HTMLInputElement;
  private speedDisplay: HTMLSpanElement;
  private timeDisplay: HTMLSpanElement;

  private timeController: TimeController;

  constructor(timeController: TimeController) {
    this.timeController = timeController;

    // Create container
    this.container = document.createElement("div");
    this.container.className = "time-control-ui";

    // Create Play/Pause button
    this.playButton = document.createElement("button");
    this.playButton.className = "time-control-button";
    this.playButton.textContent = timeController.isCurrentlyPlaying() ? "⏸ Pause" : "▶ Play";
    this.playButton.onclick = () => this.togglePlay();

    // Create speed control
    const speedContainer = document.createElement("div");
    speedContainer.className = "speed-control";

    const speedLabel = document.createElement("label");
    speedLabel.textContent = "Speed: ";

    this.speedSlider = document.createElement("input");
    this.speedSlider.type = "range";
    this.speedSlider.min = "-2"; // 10^-2 = 0.01x
    this.speedSlider.max = "0.7"; // 10^0.7 ≈ 5.0x
    this.speedSlider.step = "0.1";
    this.speedSlider.value = "0"; // 10^0 = 1.0x
    this.speedSlider.className = "speed-slider";
    this.speedSlider.oninput = () => this.updateSpeed();

    this.speedDisplay = document.createElement("span");
    this.speedDisplay.className = "speed-display";
    this.speedDisplay.textContent = "1.0x";

    speedContainer.appendChild(speedLabel);
    speedContainer.appendChild(this.speedSlider);
    speedContainer.appendChild(this.speedDisplay);

    // Create time display
    this.timeDisplay = document.createElement("div");
    this.timeDisplay.className = "time-display";
    this.timeDisplay.textContent = "Time: 0.000";

    // Assemble UI
    this.container.appendChild(this.playButton);
    this.container.appendChild(speedContainer);
    this.container.appendChild(this.timeDisplay);
  }

  /**
   * Toggle play/pause
   */
  private togglePlay(): void {
    this.timeController.togglePlay();
    this.updatePlayButton();
  }

  /**
   * Update playback speed from slider
   */
  private updateSpeed(): void {
    // Convert logarithmic slider to speed
    const logValue = parseFloat(this.speedSlider.value);
    const speed = Math.pow(10, logValue);

    this.timeController.setSpeed(speed);

    // Update display
    if (speed >= 1.0) {
      this.speedDisplay.textContent = `${speed.toFixed(1)}x`;
    } else {
      this.speedDisplay.textContent = `${speed.toFixed(2)}x`;
    }
  }

  /**
   * Update play button text
   */
  private updatePlayButton(): void {
    if (this.timeController.isCurrentlyPlaying()) {
      this.playButton.textContent = "⏸ Pause";
    } else {
      this.playButton.textContent = "▶ Play";
    }
  }

  /**
   * Update time display (call every frame)
   */
  update(): void {
    const time = this.timeController.getCurrentTime();
    this.timeDisplay.textContent = `Time: ${time.toFixed(3)}`;

    // Update play button if manual override
    if (this.timeController.isManual()) {
      this.updatePlayButton();
    }
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
