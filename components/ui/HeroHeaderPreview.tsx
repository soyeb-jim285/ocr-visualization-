"use client";

import { useState, useEffect } from "react";

const scripts = ["English", "বাংলা", "123৪৫৬", "সংযুক্ত"];

function TypingAnimation() {
  const [idx, setIdx] = useState(0);
  const [displayed, setDisplayed] = useState("");
  const [isDeleting, setIsDeleting] = useState(false);

  useEffect(() => {
    const word = scripts[idx];
    let timer: ReturnType<typeof setTimeout>;

    if (!isDeleting && displayed.length < word.length) {
      timer = setTimeout(() => setDisplayed(word.slice(0, displayed.length + 1)), 90);
    } else if (!isDeleting && displayed.length === word.length) {
      timer = setTimeout(() => setIsDeleting(true), 1400);
    } else if (isDeleting && displayed.length > 0) {
      timer = setTimeout(() => setDisplayed(displayed.slice(0, -1)), 50);
    } else {
      timer = setTimeout(() => {
        setIsDeleting(false);
        setIdx((i) => (i + 1) % scripts.length);
      }, 0);
    }

    return () => clearTimeout(timer);
  }, [displayed, isDeleting, idx]);

  return (
    <div className="h-10 sm:h-12">
      <span className="font-mono text-xl text-accent-primary sm:text-2xl md:text-3xl">
        {displayed}
        <span className="animate-pulse">|</span>
      </span>
    </div>
  );
}

export function HeroHeader() {
  return (
    <div className="flex flex-col items-center gap-1.5">
      <h1 className="text-balance text-center text-4xl font-semibold leading-[1.05] tracking-tight text-foreground sm:text-5xl md:text-6xl">
        Peel Back the Layers of Recognition
      </h1>
      <p className="max-w-xl text-center text-sm leading-relaxed text-foreground/60 sm:text-base">
        An interactive deep dive into how a CNN reads handwritten characters
        — draw anything and watch 13 layers process it in real time.
      </p>
      <TypingAnimation />
    </div>
  );
}
