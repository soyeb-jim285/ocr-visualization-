"use client";

import { useState, useEffect } from "react";
import { fetchConv1Weights } from "@/lib/model/conv1Weights";

interface Conv1WeightsState {
  kernels: number[][][] | null; // [32][3][3]
  biases: number[] | null;      // [32]
  loading: boolean;
}

export function useConv1Weights(): Conv1WeightsState {
  const [state, setState] = useState<Conv1WeightsState>({
    kernels: null,
    biases: null,
    loading: true,
  });

  useEffect(() => {
    fetchConv1Weights().then(({ kernels, biases }) => {
      setState({ kernels, biases, loading: false });
    });
  }, []);

  return state;
}
