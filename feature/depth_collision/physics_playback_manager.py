"""Physics playback manager for frame-by-frame physics simulation during playback."""

from collections import OrderedDict
from dataclasses import dataclass
import numpy as np
from typing import Optional


@dataclass
class PhysicsState:
    """Immutable physics state snapshot for caching."""
    translation: np.ndarray  # (3,) object position in local space
    velocity: np.ndarray     # (3,) velocity vector
    is_grounded: bool        # Whether object is on ground

    def copy(self):
        """Create a deep copy of this state."""
        return PhysicsState(
            translation=self.translation.copy(),
            velocity=self.velocity.copy(),
            is_grounded=self.is_grounded
        )


class PhysicsPlaybackManager:
    """
    Manages frame-by-frame physics simulation during playback.

    Key features:
    - Fixed dt per frame (1.0 / playback_fps)
    - LRU cache for physics states (100 frames default)
    - Deterministic re-simulation for backward scrubbing
    - Manual movement override support
    """

    def __init__(self, object_controller, playback_fps=60, cache_size=100):
        """
        Initialize physics playback manager.

        Args:
            object_controller: ObjectController instance
            playback_fps: Playback framerate (for calculating fixed dt)
            cache_size: Maximum number of frames to cache
        """
        self.object_controller = object_controller
        self.playback_fps = playback_fps
        self.cache_size = cache_size

        # Fixed timestep per frame
        self.fixed_dt = 1.0 / playback_fps

        # State cache: {frame_id: PhysicsState}
        self.frame_states = OrderedDict()

        # Initial state (when physics was enabled)
        self.initial_state: Optional[PhysicsState] = None
        self.initial_frame_id: Optional[int] = None

        # Last simulated frame
        self.last_simulated_frame: Optional[int] = None

        # Physics enabled flag
        self.enabled = False

    def on_physics_enabled(self, initial_frame_id):
        """
        Called when physics is enabled.

        Args:
            initial_frame_id: Frame ID when physics was enabled
        """
        self.enabled = True
        self.initial_frame_id = initial_frame_id
        self.last_simulated_frame = initial_frame_id

        # Capture initial state
        self.initial_state = self._capture_current_state()
        self.frame_states[initial_frame_id] = self.initial_state.copy()

        print(f"[PHYSICS_PLAYBACK] Enabled at frame {initial_frame_id}")
        print(f"[PHYSICS_PLAYBACK] Initial state: pos={self.initial_state.translation}, vel={self.initial_state.velocity}")

    def on_physics_disabled(self):
        """Called when physics is disabled."""
        self.enabled = False
        self.frame_states.clear()
        self.initial_state = None
        self.initial_frame_id = None
        self.last_simulated_frame = None

        print(f"[PHYSICS_PLAYBACK] Disabled, cache cleared")

    def on_frame_change(self, new_frame_id, old_frame_id, collision_detector, sphere_radius):
        """
        Called when playback frame changes.

        Args:
            new_frame_id: New frame ID
            old_frame_id: Previous frame ID
            collision_detector: DepthCollisionDetector instance
            sphere_radius: Sphere radius for collision detection
        """
        if not self.enabled or self.initial_state is None:
            return

        # Determine transition type
        frame_delta = new_frame_id - old_frame_id

        if frame_delta == 1:
            # Sequential forward: N → N+1
            self._handle_sequential_forward(new_frame_id, collision_detector, sphere_radius)
        elif frame_delta > 1:
            # Scrub forward: N → N+K
            self._handle_scrub_forward(new_frame_id, old_frame_id, collision_detector, sphere_radius)
        elif frame_delta < 0:
            # Scrub backward: N → N-K
            self._handle_scrub_backward(new_frame_id, collision_detector, sphere_radius)

    def on_manual_movement(self, current_frame_id):
        """
        Called when user manually moves object during physics playback.

        Invalidates cache from current frame onwards.

        Args:
            current_frame_id: Frame ID where manual movement occurred
        """
        if not self.enabled:
            return

        # Clear future states (they're now invalid)
        frames_to_remove = [f for f in self.frame_states.keys() if f > current_frame_id]
        for frame_id in frames_to_remove:
            del self.frame_states[frame_id]

        # Update last simulated frame
        self.last_simulated_frame = current_frame_id

        # Cache current state (with manually updated position)
        self._cache_state(current_frame_id)

        print(f"[PHYSICS_PLAYBACK] Manual movement at frame {current_frame_id}, {len(frames_to_remove)} future states invalidated")

    def clear_cache(self):
        """Clear all cached states except initial state."""
        initial_state = self.frame_states.get(self.initial_frame_id) if self.initial_frame_id else None
        self.frame_states.clear()
        if initial_state and self.initial_frame_id is not None:
            self.frame_states[self.initial_frame_id] = initial_state

        print(f"[PHYSICS_PLAYBACK] Cache cleared")

    def _handle_sequential_forward(self, new_frame_id, collision_detector, sphere_radius):
        """
        Handle sequential forward playback (N → N+1).

        Args:
            new_frame_id: New frame ID
            collision_detector: DepthCollisionDetector instance
            sphere_radius: Sphere radius
        """
        print(f"[PHYSICS_PLAYBACK] Sequential forward: {new_frame_id-1} → {new_frame_id}")

        # Simulate one step
        self._simulate_step(new_frame_id, collision_detector, sphere_radius)

    def _handle_scrub_forward(self, new_frame_id, old_frame_id, collision_detector, sphere_radius):
        """
        Handle forward scrubbing (N → N+K, K > 1).

        Args:
            new_frame_id: New frame ID
            old_frame_id: Previous frame ID
            collision_detector: DepthCollisionDetector instance
            sphere_radius: Sphere radius
        """
        print(f"[PHYSICS_PLAYBACK] Scrub forward: {old_frame_id} → {new_frame_id}")

        # Try to restore from cache
        if new_frame_id in self.frame_states:
            cached_state = self.frame_states[new_frame_id]
            self._restore_state(cached_state)
            print(f"[PHYSICS_PLAYBACK] Restored from cache")
            return

        # Cache miss: re-simulate range
        print(f"[PHYSICS_PLAYBACK] Cache miss, re-simulating range")

        # Find last cached state before new_frame_id
        last_cached_frame = None
        for frame_id in sorted(self.frame_states.keys(), reverse=True):
            if frame_id < new_frame_id:
                last_cached_frame = frame_id
                break

        if last_cached_frame is None:
            # No cached state before new_frame_id, start from initial
            last_cached_frame = self.initial_frame_id
            self._restore_state(self.initial_state)
        else:
            # Restore last cached state
            self._restore_state(self.frame_states[last_cached_frame])

        # Re-simulate range
        for frame_id in range(last_cached_frame + 1, new_frame_id + 1):
            self._simulate_step(frame_id, collision_detector, sphere_radius)

    def _handle_scrub_backward(self, new_frame_id, collision_detector, sphere_radius):
        """
        Handle backward scrubbing (N → N-K, K > 0).

        ALWAYS re-simulates from initial state for accuracy.

        Args:
            new_frame_id: New frame ID
            collision_detector: DepthCollisionDetector instance
            sphere_radius: Sphere radius
        """
        print(f"[PHYSICS_PLAYBACK] Scrub backward → {new_frame_id}, re-simulating from initial")

        # Restore initial state
        self._restore_state(self.initial_state)

        # Re-simulate entire range
        for frame_id in range(self.initial_frame_id + 1, new_frame_id + 1):
            self._simulate_step(frame_id, collision_detector, sphere_radius)

        print(f"[PHYSICS_PLAYBACK] Re-simulation complete")

    def _simulate_step(self, frame_id, collision_detector, sphere_radius):
        """
        Simulate one physics step for given frame.

        Args:
            frame_id: Frame ID to simulate
            collision_detector: DepthCollisionDetector instance
            sphere_radius: Sphere radius
        """
        # Update physics (this will modify object_controller's internal state)
        self.object_controller.update_physics(self.fixed_dt)

        # Cache result
        self._cache_state(frame_id)

        # Update last simulated frame
        self.last_simulated_frame = frame_id

    def _capture_current_state(self) -> PhysicsState:
        """
        Capture current physics state from object_controller.

        Returns:
            PhysicsState snapshot
        """
        return PhysicsState(
            translation=self.object_controller.translation.copy(),
            velocity=self.object_controller.physics_engine.velocity.copy(),
            is_grounded=self.object_controller.physics_engine.is_grounded
        )

    def _restore_state(self, state: PhysicsState):
        """
        Restore physics state to object_controller.

        Args:
            state: PhysicsState to restore
        """
        self.object_controller.translation = state.translation.copy()
        self.object_controller.physics_engine.velocity = state.velocity.copy()
        self.object_controller.physics_engine.is_grounded = state.is_grounded

    def _cache_state(self, frame_id):
        """
        Cache current state for given frame_id.

        Args:
            frame_id: Frame ID to cache
        """
        state = self._capture_current_state()
        self.frame_states[frame_id] = state

        # Enforce cache size limit (LRU eviction)
        if len(self.frame_states) > self.cache_size:
            # Remove oldest frame (first item in OrderedDict)
            oldest_frame = next(iter(self.frame_states))
            if oldest_frame != self.initial_frame_id:  # Never evict initial state
                del self.frame_states[oldest_frame]
                print(f"[PHYSICS_PLAYBACK] Cache full, evicted frame {oldest_frame}")
