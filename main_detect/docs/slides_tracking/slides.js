(() => {
  "use strict";

  const slides = Array.from(document.querySelectorAll(".slide"));
  const prevBtn = document.getElementById("prevBtn");
  const nextBtn = document.getElementById("nextBtn");
  const overviewBtn = document.getElementById("overviewBtn");
  const fullscreenBtn = document.getElementById("fullscreenBtn");
  const currentLabel = document.getElementById("currentSlide");
  const totalLabel = document.getElementById("totalSlides");
  const progressBar = document.getElementById("progressBar");
  const help = document.getElementById("help");

  let current = 0;
  let overview = false;
  let touchStartX = 0;
  let touchStartY = 0;
  let helpTimer;

  const pad = value => String(value).padStart(2, "0");

  function indexFromHash() {
    const found = window.location.hash.match(/^#slide-(\d+)$/);
    if (!found) return 0;
    return Math.min(slides.length - 1, Math.max(0, Number(found[1]) - 1));
  }

  function updateUI({ writeHash = true } = {}) {
    slides.forEach((slide, index) => {
      const active = index === current;
      slide.classList.toggle("is-active", active);
      slide.setAttribute("aria-hidden", String(!active && !overview));
    });

    currentLabel.textContent = pad(current + 1);
    totalLabel.textContent = pad(slides.length);
    progressBar.style.width = `${((current + 1) / slides.length) * 100}%`;
    prevBtn.disabled = current === 0;
    nextBtn.disabled = current === slides.length - 1;
    document.title = `${pad(current + 1)} · ${slides[current].dataset.title} — TechGAR`;

    if (writeHash) {
      history.replaceState(null, "", `#slide-${current + 1}`);
    }
  }

  function goTo(index) {
    current = Math.min(slides.length - 1, Math.max(0, index));
    updateUI();
  }

  function next() { if (!overview) goTo(current + 1); }
  function previous() { if (!overview) goTo(current - 1); }

  function setOverview(enabled) {
    overview = enabled;
    document.body.classList.toggle("is-overview", overview);
    overviewBtn.setAttribute("aria-pressed", String(overview));
    overviewBtn.title = overview ? "Thoát tổng quan (Esc)" : "Tổng quan (O)";
    updateUI({ writeHash: false });
    if (overview) {
      slides[current].scrollIntoView({ behavior: "smooth", block: "center" });
    } else {
      window.scrollTo(0, 0);
    }
  }

  async function toggleFullscreen() {
    try {
      if (!document.fullscreenElement) {
        await document.documentElement.requestFullscreen();
      } else {
        await document.exitFullscreen();
      }
    } catch (error) {
      console.warn("Trình duyệt không cho phép fullscreen:", error);
    }
  }

  function brieflyShowHelp() {
    help.classList.remove("is-hidden");
    clearTimeout(helpTimer);
    helpTimer = setTimeout(() => help.classList.add("is-hidden"), 3800);
  }

  prevBtn.addEventListener("click", previous);
  nextBtn.addEventListener("click", next);
  overviewBtn.addEventListener("click", () => setOverview(!overview));
  fullscreenBtn.addEventListener("click", toggleFullscreen);

  slides.forEach((slide, index) => {
    slide.addEventListener("click", () => {
      if (!overview) return;
      current = index;
      setOverview(false);
      updateUI();
    });
  });

  document.addEventListener("keydown", event => {
    const key = event.key;

    if (key === "Escape" && overview) {
      event.preventDefault();
      setOverview(false);
      return;
    }
    if (key.toLowerCase() === "o") {
      event.preventDefault();
      setOverview(!overview);
      return;
    }
    if (key.toLowerCase() === "f") {
      event.preventDefault();
      toggleFullscreen();
      return;
    }
    if (overview) return;

    if (["ArrowRight", "PageDown", " "].includes(key)) {
      event.preventDefault();
      next();
    } else if (["ArrowLeft", "PageUp"].includes(key)) {
      event.preventDefault();
      previous();
    } else if (key === "Home") {
      event.preventDefault();
      goTo(0);
    } else if (key === "End") {
      event.preventDefault();
      goTo(slides.length - 1);
    }
  });

  document.addEventListener("touchstart", event => {
    const touch = event.changedTouches[0];
    touchStartX = touch.clientX;
    touchStartY = touch.clientY;
  }, { passive: true });

  document.addEventListener("touchend", event => {
    if (overview) return;
    const touch = event.changedTouches[0];
    const dx = touch.clientX - touchStartX;
    const dy = touch.clientY - touchStartY;
    if (Math.abs(dx) > 55 && Math.abs(dx) > Math.abs(dy) * 1.25) {
      dx < 0 ? next() : previous();
    }
  }, { passive: true });

  window.addEventListener("hashchange", () => {
    current = indexFromHash();
    updateUI({ writeHash: false });
  });

  document.addEventListener("fullscreenchange", () => {
    fullscreenBtn.setAttribute("aria-pressed", String(Boolean(document.fullscreenElement)));
    fullscreenBtn.title = document.fullscreenElement ? "Thoát toàn màn hình (F)" : "Toàn màn hình (F)";
  });

  current = indexFromHash();
  updateUI({ writeHash: !window.location.hash });
  brieflyShowHelp();
})();
