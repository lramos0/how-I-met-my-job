// locationClassifier.js
(function (global) {
    // US state names and abbreviations for quick lookup
    const STATE_ABBR = [
        'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'
    ];
    const STATE_NAMES = [
        'alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado', 'connecticut', 'delaware', 'florida', 'georgia', 'hawaii', 'idaho', 'illinois', 'indiana', 'iowa', 'kansas', 'kentucky', 'louisiana', 'maine', 'maryland', 'massachusetts', 'michigan', 'minnesota', 'mississippi', 'missouri', 'montana', 'nebraska', 'nevada', 'new hampshire', 'new jersey', 'new mexico', 'new york', 'north carolina', 'north dakota', 'ohio', 'oklahoma', 'oregon', 'pennsylvania', 'rhode island', 'south carolina', 'south dakota', 'tennessee', 'texas', 'utah', 'vermont', 'virginia', 'washington', 'west virginia', 'wisconsin', 'wyoming', 'district of columbia'
    ];

    // Indian states/UTs to catch common CV headers
    const IN_STATE_NAMES = [
        'andhra pradesh', 'arunachal pradesh', 'assam', 'bihar', 'chhattisgarh', 'goa', 'gujarat', 'haryana', 'himachal pradesh', 'jharkhand', 'karnataka', 'kerala', 'madhya pradesh', 'maharashtra', 'manipur', 'meghalaya', 'mizoram', 'nagaland', 'odisha', 'punjab', 'rajasthan', 'sikkim', 'tamil nadu', 'telangana', 'tripura', 'uttar pradesh', 'uttarakhand', 'west bengal', 'andaman and nicobar', 'chandigarh', 'dadra and nagar haveli', 'daman and diu', 'delhi', 'jammu and kashmir', 'ladakh', 'lakshadweep', 'puducherry'
    ];

    const COUNTRY_HINTS = [
        'united states', 'usa', 'u.s.a', 'us', 'u.s.', 'canada', 'united kingdom', 'uk', 'u.k.', 'australia', 'germany', 'france', 'india', 'singapore', 'netherlands', 'ireland', 'spain', 'italy', 'brazil', 'mexico', 'china', 'japan', 'korea', 'new zealand'
    ];

    function normalizeSpaces(str) {
        return str.replace(/\s+/g, ' ').trim();
    }

    function findCityState(text) {
        // City, ST (optional ZIP)
        const cityState = text.match(/([A-Z][a-zA-Z]+(?:[\s'-][A-Z][a-zA-Z]+)*),\s?(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY|DC)(?:\s+(\d{5}(?:-\d{4})?))?/);
        if (cityState) {
            const city = cityState[1];
            const st = cityState[2];
            const zip = cityState[3] ? ` ${cityState[3]}` : '';
            return `${city}, ${st}${zip}`;
        }
        return null;
    }

    function findStateName(text) {
        for (let i = 0; i < STATE_NAMES.length; i++) {
            const name = STATE_NAMES[i];
            const re = new RegExp(`\\b${name}\\b`, 'i');
            const m = text.match(re);
            if (m) return m[0].replace(/\b\w/g, c => c.toUpperCase());
        }
        return null;
    }

    function findIndianState(text) {
        for (let i = 0; i < IN_STATE_NAMES.length; i++) {
            const name = IN_STATE_NAMES[i];
            const re = new RegExp(`\\b${name}\\b`, 'i');
            const m = text.match(re);
            if (m) return m[0].replace(/\b\w/g, c => c.toUpperCase());
        }
        return null;
    }

    function findZip(text) {
        // US ZIP or ZIP+4, or India PIN (6 digits, optional space)
        const m = text.match(/\b(\d{5}(?:-\d{4})?|\d{3}\s?\d{3})\b/);
        return m ? m[0] : null;
    }

    function findCountry(text) {
        for (const c of COUNTRY_HINTS) {
            const re = new RegExp(`\\b${c}\\b`, 'i');
            if (re.test(text)) return c.replace(/\b\w/g, ch => ch.toUpperCase());
        }
        return null;
    }

    function extractLocation(text) {
        const raw = text || '';
        if (!raw.trim()) return 'Unknown';

        // Prioritize the top portion of the resume where contact info often sits
        const headerSlice = raw.slice(0, 1200);

        // Try City, ST, ZIP first (most specific)
        let loc = findCityState(headerSlice) || findCityState(raw);
        if (loc) return normalizeSpaces(loc);

        // Try US state names
        const stateName = findStateName(headerSlice) || findStateName(raw);
        if (stateName) {
            const zipNear = findZip(headerSlice) || findZip(raw);
            if (zipNear) return normalizeSpaces(`${stateName} ${zipNear}`);
            return normalizeSpaces(stateName);
        }

        // Try Indian state names
        const inState = findIndianState(headerSlice) || findIndianState(raw);
        if (inState) {
            const pin = findZip(headerSlice) || findZip(raw);
            if (pin) return normalizeSpaces(`${inState} ${pin}`);
            return normalizeSpaces(inState);
        }

        // Try ZIP alone
        const zip = findZip(headerSlice) || findZip(raw);
        if (zip) return zip;

        // Try country hints
        const country = findCountry(headerSlice) || findCountry(raw);
        if (country) return country;

        return 'Unknown';
    }

    // Expose globally for script.js
    global.extractLocation = extractLocation;
})(window);
