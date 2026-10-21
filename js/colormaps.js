/**
 * Color mapping utilities for activation maps and Grad-CAM heatmaps.
 * Maps normalized values [0, 1] to [R, G, B] triplets.
 */

export const COLORMAPS = {
  JET: 'jet',
  TURBO: 'turbo',
  VIRIDIS: 'viridis',
  INFERNO: 'inferno',
  CYAN: 'cyan',
  GRAY: 'gray'
};

export function getRGB(val, colormap = COLORMAPS.JET) {
  const v = Math.max(0.0, Math.min(1.0, val));

  switch (colormap) {
    case COLORMAPS.JET: {
      // Classic MATLAB/OpenCV Jet colormap (Blue -> Cyan -> Green -> Yellow -> Red)
      const fourV = 4 * v;
      const r = Math.min(fourV - 1.5, -fourV + 4.5);
      const g = Math.min(fourV - 0.5, -fourV + 3.5);
      const b = Math.min(fourV + 0.5, -fourV + 2.5);
      return [
        Math.round(Math.max(0, Math.min(1, r)) * 255),
        Math.round(Math.max(0, Math.min(1, g)) * 255),
        Math.round(Math.max(0, Math.min(1, b)) * 255)
      ];
    }

    case COLORMAPS.TURBO: {
      // Google Turbo Colormap (smooth, perceptually uniform)
      const r = 0.1357 + v * (4.5974 - v * (42.3277 - v * (130.5887 - v * (150.5667 - v * 58.1375))));
      const g = 0.0914 + v * (2.1856 + v * (4.8052 - v * (14.0195 - v * (4.2109 - v * 2.7747))));
      const b = 0.1067 + v * (12.5732 - v * (80.1472 - v * (205.8194 - v * (234.3308 - v * 101.4468))));
      return [
        Math.round(Math.max(0, Math.min(1, r)) * 255),
        Math.round(Math.max(0, Math.min(1, g)) * 255),
        Math.round(Math.max(0, Math.min(1, b)) * 255)
      ];
    }

    case COLORMAPS.VIRIDIS: {
      // Purple -> Teal -> Green -> Yellow
      if (v < 0.35) {
        const t = v / 0.35;
        return [Math.round(68 - t * 20), Math.round(1 + t * 100), Math.round(84 + t * 60)];
      } else if (v < 0.7) {
        const t = (v - 0.35) / 0.35;
        return [Math.round(48 - t * 15), Math.round(101 + t * 75), Math.round(144 - t * 40)];
      } else {
        const t = (v - 0.7) / 0.3;
        return [Math.round(33 + t * 220), Math.round(176 + t * 50), Math.round(104 - t * 70)];
      }
    }

    case COLORMAPS.INFERNO: {
      // Black -> Purple -> Orange -> Bright Yellow
      if (v < 0.33) {
        const t = v / 0.33;
        return [Math.round(t * 120), 0, Math.round(t * 180)];
      } else if (v < 0.66) {
        const t = (v - 0.33) / 0.33;
        return [Math.round(120 + t * 135), Math.round(t * 130), Math.round(180 - t * 150)];
      } else {
        const t = (v - 0.66) / 0.34;
        return [255, Math.round(130 + t * 125), Math.round(30 + t * 225)];
      }
    }

    case COLORMAPS.CYAN: {
      // Dark slate -> Neon Cyan (#00f5d4) -> Bright White
      if (v < 0.8) {
        const t = v / 0.8;
        return [
          Math.round(10 + t * (0 - 10)),
          Math.round(18 + t * (245 - 18)),
          Math.round(35 + t * (212 - 35))
        ];
      } else {
        const t = (v - 0.8) / 0.2;
        return [
          Math.round(t * 255),
          Math.round(245 + t * 10),
          Math.round(212 + t * 43)
        ];
      }
    }

    case COLORMAPS.GRAY:
    default: {
      const g = Math.round(v * 255);
      return [g, g, g];
    }
  }
}
