/**
 * Application outcome survey — GradCafe-style flow: pick Fortune 500 company,
 * profile (manual or resume text), log outcome. Stored locally only.
 */
(function () {
  var STORAGE_KEY = "himmj_outcomes_v1";

  function $(id) {
    return document.getElementById(id);
  }

  function debounce(fn, ms) {
    var t;
    return function () {
      var a = arguments;
      clearTimeout(t);
      t = setTimeout(function () {
        fn.apply(null, a);
      }, ms);
    };
  }

  /** Insert generated SVG (avoids data-URL # fragment issues with <img src>). */
  function mountIcon(el, company, size) {
    if (!el || !window.Fortune500 || !company) return;
    el.innerHTML = window.Fortune500.iconMarkSvg(company, size);
    var svg = el.querySelector("svg");
    if (svg) {
      svg.setAttribute("width", String(size));
      svg.setAttribute("height", String(size));
      svg.style.display = "block";
      svg.style.flexShrink = "0";
    }
  }

  /** Best-effort extraction from pasted resume / plain text */
  function parseResumeText(text) {
    var out = { fullName: "", email: "", phone: "", rawSnippet: "" };
    if (!text || !text.trim()) return out;
    var lines = text.split(/\r?\n/).map(function (l) {
      return l.trim();
    });
    var emailRe = /[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}/i;
    var phoneRe = /(\+?\d[\d\s().-]{8,}\d)/;
    var m = text.match(emailRe);
    if (m) out.email = m[0];
    var p = text.match(phoneRe);
    if (p) out.phone = p[1].replace(/\s+/g, " ").trim();
    for (var i = 0; i < Math.min(lines.length, 12); i++) {
      if (
        lines[i] &&
        lines[i].length > 3 &&
        lines[i].length < 80 &&
        !emailRe.test(lines[i]) &&
        /^[A-Za-z\s.'-]+$/.test(lines[i])
      ) {
        out.fullName = lines[i];
        break;
      }
    }
    out.rawSnippet = text.slice(0, 400).trim();
    return out;
  }

  function loadEntries() {
    try {
      var j = localStorage.getItem(STORAGE_KEY);
      return j ? JSON.parse(j) : [];
    } catch (e) {
      return [];
    }
  }

  function saveEntry(entry) {
    var all = loadEntries();
    entry.id = "o-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 8);
    entry.savedAt = new Date().toISOString();
    all.unshift(entry);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(all.slice(0, 200)));
    return entry;
  }

  var pickedCompany = null;
  var els = {};

  function cacheEls() {
    els.dialog = $("outcomeSurveyDialog");
    els.closeBtn = $("surveyCloseBtn");
    els.openBtns = document.querySelectorAll("[data-open-outcome-survey]");
    els.companySearch = $("surveyCompanySearch");
    els.results = $("surveyCompanyResults");
    els.picked = $("surveyPicked");
    els.pickedIcon = $("surveyPickedIcon");
    els.pickedName = $("surveyPickedName");
    els.pickedMeta = $("surveyPickedMeta");
    els.fullName = $("surveyFullName");
    els.email = $("surveyEmail");
    els.phone = $("surveyPhone");
    els.resumeFile = $("surveyResumeFile");
    els.resumePaste = $("surveyResumePaste");
    els.dropZone = $("surveyDropZone");
    els.outcome = $("surveyOutcome");
    els.dateApplied = $("surveyDateApplied");
    els.notes = $("surveyNotes");
    els.submit = $("surveySubmitBtn");
    els.success = $("surveySuccess");
    els.formBlock = $("surveyFormBlock");
  }

  function renderResults(companies) {
    if (!els.results) return;
    els.results.innerHTML = "";
    if (!companies.length) return;
    var ul = document.createElement("ul");
    companies.forEach(function (c) {
      var li = document.createElement("li");
      var btn = document.createElement("button");
      btn.type = "button";
      var iconSlot = document.createElement("span");
      iconSlot.className = "survey-icon-slot";
      mountIcon(iconSlot, c, 36);
      var span = document.createElement("span");
      span.textContent = c.name;
      var meta = document.createElement("span");
      meta.className = "r-meta";
      meta.textContent = "#" + c.rank;
      btn.appendChild(iconSlot);
      btn.appendChild(span);
      btn.appendChild(meta);
      btn.addEventListener("click", function () {
        selectCompany(c);
      });
      li.appendChild(btn);
      ul.appendChild(li);
    });
    els.results.appendChild(ul);
  }

  function selectCompany(c) {
    pickedCompany = c;
    if (els.companySearch) els.companySearch.value = c.name;
    if (els.results) els.results.innerHTML = "";
    if (els.picked) els.picked.classList.remove("hidden");
    if (els.pickedIcon) {
      mountIcon(els.pickedIcon, c, 44);
      els.pickedIcon.setAttribute("title", c.name);
    }
    if (els.pickedName) els.pickedName.textContent = c.name;
    if (els.pickedMeta)
      els.pickedMeta.textContent =
        "Fortune " +
        c.rank +
        " · " +
        c.domain +
        " · $" +
        Math.round(c.revenue / 1000) +
        "B rev (FY list year)";
  }

  var doSearch = debounce(function () {
    var q = els.companySearch && els.companySearch.value ? els.companySearch.value.trim() : "";
    if (q.length < 1) {
      if (els.results) els.results.innerHTML = "";
      return;
    }
    var list = window.Fortune500.searchCompanies(q, 20);
    renderResults(list);
  }, 180);

  function handleResumeFile(file) {
    if (!file) return;
    var ext = (file.name.split(".").pop() || "").toLowerCase();
    if (ext === "txt" || ext === "md") {
      var reader = new FileReader();
      reader.onload = function () {
        var t = String(reader.result || "");
        applyParsed(parseResumeText(t));
        if (els.resumePaste && !els.resumePaste.value) els.resumePaste.value = t.slice(0, 8000);
      };
      reader.readAsText(file);
      return;
    }
    if (els.resumePaste) {
      els.resumePaste.placeholder =
        "Paste plain text from your resume here (PDF/DOCX upload is browser-limited — export or copy text).";
    }
  }

  function applyParsed(p) {
    if (p.fullName && els.fullName && !els.fullName.value) els.fullName.value = p.fullName;
    if (p.email && els.email && !els.email.value) els.email.value = p.email;
    if (p.phone && els.phone && !els.phone.value) els.phone.value = p.phone;
  }

  function resetForm() {
    pickedCompany = null;
    if (els.picked) els.picked.classList.add("hidden");
    [
      "surveyCompanySearch",
      "surveyFullName",
      "surveyEmail",
      "surveyPhone",
      "surveyResumePaste",
      "surveyNotes",
    ].forEach(function (id) {
      var e = $(id);
      if (e) e.value = "";
    });
    if (els.results) els.results.innerHTML = "";
    if (els.resumeFile) els.resumeFile.value = "";
    if (els.dateApplied) els.dateApplied.valueAsDate = new Date();
    if (els.success) els.success.classList.add("hidden");
    if (els.formBlock) els.formBlock.style.display = "";
  }

  function openSurvey() {
    resetForm();
    window.Fortune500.ready().then(
      function () {
        if (els.dialog && typeof els.dialog.showModal === "function") els.dialog.showModal();
      },
      function () {
        alert("Could not load the Fortune 500 list. Ensure data/fortune500.csv is present and you are serving over HTTP.");
      },
    );
  }

  function closeSurvey() {
    if (els.dialog && typeof els.dialog.close === "function") els.dialog.close();
  }

  function onSubmit(e) {
    e.preventDefault();
    if (!pickedCompany) {
      alert("Choose a company from the Fortune 500 search results.");
      return;
    }
    var entry = {
      companyRank: pickedCompany.rank,
      companyName: pickedCompany.name,
      companyDomain: pickedCompany.domain,
      fullName: els.fullName ? els.fullName.value.trim() : "",
      email: els.email ? els.email.value.trim() : "",
      phone: els.phone ? els.phone.value.trim() : "",
      outcome: els.outcome ? els.outcome.value : "",
      dateApplied: els.dateApplied ? els.dateApplied.value : "",
      notes: els.notes ? els.notes.value.trim() : "",
      resumeExcerpt: els.resumePaste ? els.resumePaste.value.trim().slice(0, 1200) : "",
    };
    saveEntry(entry);
    if (els.formBlock) els.formBlock.style.display = "none";
    if (els.success) {
      els.success.classList.remove("hidden");
      els.success.textContent =
        "Saved locally — thank you for contributing an anonymous outcome for " + pickedCompany.name + ".";
    }
  }

  function bind() {
    cacheEls();
    if (els.openBtns) {
      els.openBtns.forEach(function (b) {
        b.addEventListener("click", openSurvey);
      });
    }
    if (els.closeBtn) els.closeBtn.addEventListener("click", closeSurvey);
    if (els.companySearch) els.companySearch.addEventListener("input", doSearch);
    if (els.submit) els.submit.addEventListener("click", onSubmit);
    if (els.resumeFile) {
      els.resumeFile.addEventListener("change", function () {
        handleResumeFile(els.resumeFile.files && els.resumeFile.files[0]);
      });
    }
    if (els.resumePaste) {
      els.resumePaste.addEventListener(
        "blur",
        debounce(function () {
          applyParsed(parseResumeText(els.resumePaste.value));
        }, 300),
      );
    }
    if (els.dropZone && els.resumeFile) {
      ["dragover", "dragenter"].forEach(function (ev) {
        els.dropZone.addEventListener(ev, function (e) {
          e.preventDefault();
          els.dropZone.classList.add("dragover");
        });
      });
      ["dragleave", "drop"].forEach(function (ev) {
        els.dropZone.addEventListener(ev, function (e) {
          e.preventDefault();
          els.dropZone.classList.remove("dragover");
        });
      });
      els.dropZone.addEventListener("drop", function (e) {
        var f = e.dataTransfer.files && e.dataTransfer.files[0];
        if (f) handleResumeFile(f);
      });
      els.dropZone.addEventListener("click", function () {
        els.resumeFile.click();
      });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bind);
  } else {
    bind();
  }
})();
