// netlify/functions/get-jobs.js
const { createClient } = require('@supabase/supabase-js');

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_KEY = process.env.SUPABASE_KEY;

exports.handler = async (event, context) => {
  try {
    const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

    // --- Parse industries from query string or body ---
    let industries = [];

    // GET: /get-jobs?industries=data%20science,analytics
    if (event.queryStringParameters && event.queryStringParameters.industries) {
      industries = event.queryStringParameters.industries
        .split(',')
        .map(part => decodeURIComponent(part).trim())
        .filter(Boolean);
    }

    // (Optional) also support POST with JSON body:
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
      .order('job_posted_date', { ascending: false }) // optional but nice
      .limit(250);

    // If we have industries, filter by them
    if (industries.length > 0) {
      // Example industries: ["Data", "Analytics"]
      // Build OR conditions like:
      //   job_industries.ilike.%Data%,job_industries.ilike.%Analytics%
      const orClauses = industries.map(industry => {
        // Escape % and _ which are special in LIKE
        const safe = industry.replace(/[%_]/g, m => '\\' + m);
        return `job_industries.ilike.%${safe}%`;
      });

      query = query.or(orClauses.join(',')); // OR across industries
    }

    const { data, error } = await query;

    if (error) {
      console.error('Supabase error:', error);
      return {
        statusCode: 500,
        body: JSON.stringify({ error: 'Supabase error', details: error.message }),
      };
    }

    return {
      statusCode: 200,
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data || []),
    };
  } catch (error) {
    console.error('Error fetching jobs:', error);
    return {
      statusCode: 500,
      body: JSON.stringify({ error: 'Failed to fetch jobs: ' + error.message }),
    };
  }
};

