const { createClient } = require('@supabase/supabase-js');

// --- CONFIGURATION ---
const SUPABASE_URL = process.env.SUPABASE_URL
const SUPABASE_KEY = process.env.SUPABASE_KEY

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
