import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { GraphPosition } from "../types";

type LayoutStore = {
  layouts: Record<string, Record<string, GraphPosition>>;
  setLayout: (key: string, positions: Record<string, GraphPosition>) => void;
  moveNodes: (key: string, positions: Record<string, GraphPosition>) => void;
};

export const useGraphLayouts = create<LayoutStore>()(
  persist(
    (set) => ({
      layouts: {},
      setLayout: (key, positions) =>
        set((state) => ({
          layouts: Object.fromEntries(
            [
              ...Object.entries(state.layouts).filter(([id]) => id !== key),
              [key, positions],
            ].slice(-30),
          ),
        })),
      moveNodes: (key, positions) =>
        set((state) => ({
          layouts: {
            ...state.layouts,
            [key]: { ...state.layouts[key], ...positions },
          },
        })),
    }),
    {
      name: "kollektiv-graph-layouts",
      partialize: (state) => ({ layouts: state.layouts }),
    },
  ),
);
