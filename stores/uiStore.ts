import { create } from "zustand";

export type HeroStage = "drawing" | "shrinking" | "revealed";

interface UIState {
  activeSection: number; // 0-9 section index
  scrollProgress: number; // 0-1 overall page scroll
  modelLoaded: boolean;
  heroStage: HeroStage;

  setActiveSection: (idx: number) => void;
  setScrollProgress: (val: number) => void;
  setModelLoaded: (val: boolean) => void;
  setHeroStage: (stage: HeroStage) => void;
}

export const useUIStore = create<UIState>((set) => ({
  activeSection: 0,
  scrollProgress: 0,
  modelLoaded: false,
  heroStage: "drawing",

  setActiveSection: (idx) => set({ activeSection: idx }),
  setScrollProgress: (val) => set({ scrollProgress: val }),
  setModelLoaded: (val) => set({ modelLoaded: val }),
  setHeroStage: (stage) => set({ heroStage: stage }),
}));
