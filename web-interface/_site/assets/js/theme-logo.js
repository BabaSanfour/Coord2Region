(function () {
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)");

  function isDarkMode() {
    const explicit = document.documentElement.getAttribute("data-theme");
    if (explicit === "light" || explicit === "dark") {
      return explicit === "dark";
    }
    return prefersDark.matches;
  }

  function updateLogos() {
    const dark = isDarkMode();
    document.querySelectorAll(".theme-logo").forEach((img) => {
      const wrapper = img.closest(".logo-text-block");
      if (!wrapper) return;
      const lightSrc = wrapper.dataset.logoLight;
      const darkSrc = wrapper.dataset.logoDark;
      const nextSrc = dark ? darkSrc : lightSrc;
      if (nextSrc && img.getAttribute("src") !== nextSrc) {
        img.setAttribute("src", nextSrc);
      }
    });
  }

  updateLogos();
  prefersDark.addEventListener("change", updateLogos);

  const observer = new MutationObserver((mutations) => {
    for (const m of mutations) {
      if (m.type === "attributes" && m.attributeName === "data-theme") {
        updateLogos();
        break;
      }
    }
  });
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
})();
