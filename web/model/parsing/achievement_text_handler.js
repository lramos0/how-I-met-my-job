// achievementClassifier.js
(function (global) {
    function pushOnce(arr, value) {
        if (value && !arr.includes(value)) arr.push(value);
    }

    function extractAchievements(text) {
        const raw = text || "";
        const lower = raw.toLowerCase();
        const achievements = [];

        // Dean / Chancellor honors
        if (/dean['’]?s\s+list/.test(lower)) pushOnce(achievements, "Dean's List");
        if (/chancellor['’]?s\s+list/.test(lower)) pushOnce(achievements, "Chancellor's List");

        // Latin honors
        if (/summa\s+cum\s+laude/.test(lower)) pushOnce(achievements, "Summa Cum Laude");
        if (/magna\s+cum\s+laude/.test(lower)) pushOnce(achievements, "Magna Cum Laude");
        if (/cum\s+laude/.test(lower)) pushOnce(achievements, "Cum Laude");

        // Scholarships & fellowships
        if (/scholarship/.test(lower)) pushOnce(achievements, "Scholarship Recipient");
        if (/fellowship/.test(lower)) pushOnce(achievements, "Fellowship Recipient");

        // Publications or best paper awards
        if (/(best\s+paper|best\s+poster|publication\s+award)/.test(lower)) pushOnce(achievements, "Best Paper/Poster Award");

        // Certifications framed as achievements (avoid duplicates with cert parsing)
        if (/(certificate of merit|merit certificate)/.test(lower)) pushOnce(achievements, "Certificate of Merit");

        // Award / prize / honor details (capture the phrase so we keep the context)
        const sentenceSplit = /[.!?\n]+/;
        const sentences = raw.split(sentenceSplit).map(s => s.trim()).filter(Boolean);
        const awardish = /(award|awarded|prize|honor|honour|recognition|recipient|won|winner|champion|1st\s*place|first\s*place|runner[-\s]?up|finalist|semi[-\s]?finalist|top\s*\d+%?|hackathon|competition|contest|case competition|datathon|codefest|codeathon|kaggle|leetcode|codeforces|icpc|acm|olympiad|math contest|science fair|design challenge|robotics)/i;
        sentences.forEach(s => {
            if (awardish.test(s)) {
                const trimmed = s.length > 180 ? s.slice(0, 177) + '…' : s;
                pushOnce(achievements, trimmed);
            }
        });

        // Additional targeted capture for short fragments after keywords when no sentence split is present
        const fragmentRegex = /(?:won|awarded|received|earned|secured|placed|runner[-\s]?up|finalist)[:\-]?\s{0,3}([\w\s,&()\-]{3,80})/gi;
        let m;
        while ((m = fragmentRegex.exec(raw)) !== null) {
            const detail = m[0].trim();
            if (detail) pushOnce(achievements, detail);
        }

        return achievements;
    }

    // Expose globally for script.js
    global.extractAchievements = extractAchievements;
})(window);
