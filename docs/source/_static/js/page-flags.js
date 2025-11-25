(function () {
  function setPageFlag() {
    if (typeof DOCUMENTATION_OPTIONS === "undefined") {
      return;
    }
    var name = DOCUMENTATION_OPTIONS.pagename || "";
    if (!name || !document.body) {
      return;
    }
    document.body.dataset.page = name;
  }

  function collapseDocsSections() {
    var sidebars = document.querySelectorAll(".bd-sidebar nav");
    if (!sidebars.length) {
      return;
    }

    var targets = [
      "documentation/tutorials.html",
      "auto_examples/index.html",
    ];

    sidebars.forEach(function (nav) {
      targets.forEach(function (hrefSuffix) {
        var links = nav.querySelectorAll(
          'a.reference.internal[href$="' + hrefSuffix + '"]',
        );
        links.forEach(function (link) {
          var parent = link.closest("li.has-children");
          if (!parent || parent.classList.contains("current")) {
            return;
          }
          var details = parent.querySelector("details");
          if (details) {
            details.removeAttribute("open");
          }
        });
      });
    });
  }

  function boot() {
    setPageFlag();
    collapseDocsSections();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
