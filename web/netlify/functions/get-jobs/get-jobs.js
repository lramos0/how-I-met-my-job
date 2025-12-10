// netlify/functions/get-jobs.js
const { createClient } = require('@supabase/supabase-js');

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_KEY = process.env.SUPABASE_KEY;

exports.handler = async (event, context) => {
  try {
    const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

    // --- Parse industries from query string or body ---
    let industries = [];

    if (event.queryStringParameters && event.queryStringParameters.industries) {
      industries = event.queryStringParameters.industries
        .split(',')
        .map(part => decodeURIComponent(part).trim())
        .filter(Boolean);
    }

    if (!industries.length && event.httpMethod === 'POST' && event.body) {
      try {
        const body = JSON.parse(event.body);
        if (Array.isArray(body.industries)) {
          industries = body.industries
            .map(s => String(s).trim())
            .filter(Boolean);
        }
      } catch (e) {
        console.warn('Failed to parse JSON body:', e);
      }
    }

    // --- Build Supabase query ---
    let query = supabase
      .from('job_listings')
      .select('*')
      .order('job_posted_date', { ascending: false }) // newest first
      .limit(1000);

    if (industries.length > 0) {
      const orClauses = industries.map(industry => {
        const safe = industry.replace(/[%_]/g, m => '\\' + m);
        return `job_industries.ilike.%${safe}%`;
      });

      query = query.or(orClauses.join(','));
    }

    const { data, error } = await query;

    if (error) {
      console.error('Supabase error:', error);
      return {
        statusCode: 500,
        body: JSON.stringify({ error: 'Supabase error', details: error.message }),
      };
    }

    // --- Enforce uniqueness on job_title ---
    const seenTitles = new Set();
    const uniqueJobs = [];

    for (const row of data || []) {
      // normalize title to avoid "Data Scientist" vs "data scientist" double-counts
      const titleKey = (row.job_title || '').trim().toLowerCase();
      if (!titleKey) continue;

      if (seenTitles.has(titleKey)) continue;
      seenTitles.add(titleKey);
      uniqueJobs.push(row);
    }

    return {
      statusCode: 200,
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(uniqueJobs),
    };
  } catch (error) {
    console.error('Error fetching jobs:', error);
    return {
      statusCode: 500,
      body: JSON.stringify({ error: 'Failed to fetch jobs: ' + error.message }),
    };
  }
};


