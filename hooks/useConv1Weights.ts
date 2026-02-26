"use client";

import { useState, useEffect } from "react";
import { fetchConv1Weights } from "@/lib/model/conv1Weights";

interface Conv1WeightsState {
  kernels: number[][][] | null;
  biases: number[] | null;
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
