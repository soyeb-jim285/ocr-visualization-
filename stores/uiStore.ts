import { create } from "zustand";

export type ViewMode = "2d" | "3d";
export type HeroStage = "drawing" | "shrinking" | "revealed";

interface UIState {
  viewMode: ViewMode;
  activeSection: number; // 0-9 section index
  scrollProgress: number; // 0-1 overall page scroll
  modelLoaded: boolean;
  modelLoadingProgress: number; // 0-1
  heroStage: HeroStage;

  setViewMode: (mode: ViewMode) => void;
  setActiveSection: (idx: number) => void;
  setScrollProgress: (val: number) => void;
  setModelLoaded: (val: boolean) => void;
  setModelLoadingProgress: (val: number) => void;
  setHeroStage: (stage: HeroStage) => void;
}

export const useUIStore = create<UIState>((set) => ({
  viewMode: "2d",
  activeSection: 0,
  scrollProgress: 0,
  modelLoaded: false,
  modelLoadingProgress: 0,
  heroStage: "drawing",

  setViewMode: (mode) => set({ viewMode: mode }),
  setActiveSection: (idx) => set({ activeSection: idx }),
  setScrollProgress: (val) => set({ scrollProgress: val }),
  setModelLoaded: (val) => set({ modelLoaded: val }),
  setModelLoadingProgress: (val) => set({ modelLoadingProgress: val }),
  setHeroStage: (stage) => set({ heroStage: stage }),
}));
