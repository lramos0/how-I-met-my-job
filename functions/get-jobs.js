const { createClient } = require('@supabase/supabase-js');

// --- CONFIGURATION ---
const SUPABASE_URL = process.env.SUPABASE_URL || 'https://xaooqthrquigpwpsbmss.supabase.co';
const SUPABASE_KEY = process.env.SUPABASE_KEY || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inhhb29xdGhycXVpZ3B3cHNibXNzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM1NTM3MDYsImV4cCI6MjA3OTEyOTcwNn0.nQKw857xscGOew4mNrrn_AIb3EHLq84x9j3BWMR3Lds';

exports.handler = async (event, context) => {
    try {
        const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

        // Gets 50 jobs
        const { data, error } = await supabase
            .from('job_listings')
            .select('*')
            .limit(50);

        if (error) {
            console.error("Supabase error:", error);
            throw error;
        }

        return {
            statusCode: 200,
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        };
    } catch (error) {
        console.error("Error fetching jobs:", error);
        return {
            statusCode: 500,
            body: JSON.stringify({ error: "Failed to fetch jobs: " + error.message })
        };
    }
};
