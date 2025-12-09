// educationClassifier.js
window.extractEducation = function (text) {
    text = text.toLowerCase();

    // ============================================================
    //                       PH.D / DOCTORAL
    // ============================================================
    const phdRegex = [
        /\bph\.?\s*d\.?\b/,                                   // PhD / Ph.D.
        /\bdoctorate\b/,
        /\bdoctor of\b/,
        /\bdoctors in\b/,
        /\bdphil\b/,
        /\bsc\.?\s*d\.?\b/,                                   // ScD

        // Law / JD
        /\bj\.?\s*d\.?\b/,
        /\bjuris doctor\b/,
        /\blaw school\b/,

        // Medical / Clinical doctor-level
        /\bdo\b(?!\b.*not)/,                                  // DO (doctor of osteopathy)
        /\bd\.?\s*d\.?\s*s\.?\b/,                             // DDS
        /\bd\.?\s*m\.?\s*s\.?\b/,                             // DMS
        /\bd\.?\s*n\.?\s*p\.?\b/,                             // DNP
        /\bpharmd\b/,                                         // PharmD
        /\bpsy\.?\s*d\.?\b|\bpsychology doctorate\b/,
        /\bjunior doctor\b/
    ];

    if (phdRegex.some(r => r.test(text))) {
        return "Doctorate";
    }

    // ============================================================
    //                          MASTER
    // ============================================================
    const masterPositive = [
        /\bmaster['’]s\b/,
        /\bmaster of\b/,

        /\bms\b|\bm\.?\s*s\.?\b/,
        /\bma\b|\bm\.?\s*a\.?\b/,
        /\bmsc\b/,
        /\bmba\b/,
        /\bmeng\b|\bm\.?\s*eng\.?\b/,
        /\bm\.?ed\.?\b/,

        // Nursing
        /\bm\.?\s*s\.?\s*n\.?\b/,     // MSN
        /\bnurse practitioner\b/,     // NP usually masters-level
    ];

    const masterNegative = [
        /\b(master bedroom|master plan|master key)\b/,
        /\b(headmaster|grandmaster|master electrician)\b/
    ];

    if (
        masterPositive.some(r => r.test(text)) &&
        !masterNegative.some(r => r.test(text))
    ) {
        return "Master";
    }

    // ============================================================
    //                          BACHELOR
    // ============================================================
    const bachelorPositive = [
        /\bbachelor['’]s\b/,
        /\bbachelor of\b/,

        /\bbs\b|\bb\.?\s*s\.?\b/,
        /\bba\b|\bb\.?\s*a\.?\b/,
        /\bbsc\b/,
        /\bbfa\b/,
        /\bbeng\b/,

        /\bbe\b|\bb\.?\s*e\.?\b/,

        // Nursing
        /\bbsn\b/, // Bachelor of Science in Nursing
    ];

    const bachelorNegative = [
        /\bbachelor party\b/,
        /\bbachelor pad\b/
    ];

    if (
        bachelorPositive.some(r => r.test(text)) &&
        !bachelorNegative.some(r => r.test(text))
    ) {
        return "Bachelor";
    }

    // ============================================================
    //                    ASSOCIATES DEGREE
    // ============================================================
    const associateRegex = [
        /\bassociate['’]s\b/,
        /\bassociate of\b/,
        /\baa\b\b|\ba\.?\s*a\.?\b/,
        /\bas\b\b|\ba\.?\s*s\.?\b/,
        /\baas\b/,      // Associate of Applied Science
        /\baos\b/,      // Associate of Occupational Studies
    ];

    if (associateRegex.some(r => r.test(text))) {
        return "Associate";
    }

    // ============================================================
    //                      NURSING PROGRAMS
    // ============================================================
    const nursingRegex = [
        /\brn\b(?!\w)/,                 // RN
        /\blicensed practical nurse\b/,
        /\blicensed vocational nurse\b/,
        /\blpn\b|\blvn\b/,
        /\bcna\b/,                      // Certified Nursing Assistant
        /\bpnp\b/,                      // Pediatric NP (if not classified above)
        /\baprn\b/,                     // advanced practice nurse
    ];

    if (nursingRegex.some(r => r.test(text))) {
        return "Nursing Program";
    }

    // ============================================================
    //                          TRADE SCHOOL
    // ============================================================
    const tradeSchoolRegex = [
        /\btrade school\b/,
        /\btechnical school\b/,
        /\btech school\b/,
        /\bcommunity college certificate\b/,

        // Trade occupations
        /\bhvac\b/,
        /\bwelding\b|\bweld(er|ing)\b/,
        /\belectrician\b/,
        /\bplumbing\b|\bplumber\b/,
        /\bcarpentry\b|\bcarpenter\b/,
        /\bautomotive\b/,
        /\bmechanic training\b/,
        /\bcosmetology\b/,
        /\bculinary school\b/,
    ];

    if (tradeSchoolRegex.some(r => r.test(text))) {
        return "Trade School / Vocational";
    }

    // ============================================================
    //                  MILITARY EDUCATION / TRAINING
    // ============================================================
    const militaryRegex = [
        /\bmilitary training\b/,
        /\barmy education\b/,
        /\bnavy school\b/,
        /\bair force academy\b/,
        /\bmarine corps school\b/,
        /\bcoast guard school\b/,
        /\bboot camp\b/,
        /\bmos school\b/,             // Army MOS training
        /\brate school\b/,            // Navy rate training
        /\bmilitary occupational specialty\b/,
    ];

    if (militaryRegex.some(r => r.test(text))) {
        return "Military Education / Training";
    }

    // ============================================================
    //                  CERTIFICATIONS / TRAINING PROGRAMS
    // ============================================================
    const certificateRegex = [
        /\bcertificate\b/,
        /\bcertification\b/,
        /\bcertified\b/,
        /\btraining program\b/,
        /\bprofessional training\b/,
        /\bcontinuing education\b/,
        /\bbootcamp\b/,
        /\bworkshop\b/,
    ];

    if (certificateRegex.some(r => r.test(text))) {
        return "Certificate / Training Program";
    }

    return "Unknown";
};
