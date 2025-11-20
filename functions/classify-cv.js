exports.handler = async (event) => {
    try {
        const body = event.body ? JSON.parse(event.body) : {};

        // Simple password check (keeps parity with frontend)
        if (!body.password || body.password !== 'craig123') {
            return {
                statusCode: 401,
                body: JSON.stringify({ error: 'Unauthorized: invalid password' })
            };
        }

        const inputs = Array.isArray(body.inputs) ? body.inputs : [];

        // Create a simple mocked prediction for each input
        const predictions = inputs.map((input) => ({
            candidate_id: input.candidate_id || null,
            competitive_score: Math.round(Math.random() * 100),
            matched_roles: []
        }));

        return {
            statusCode: 200,
            body: JSON.stringify({ predictions })
        };
    } catch (err) {
        return {
            statusCode: 500,
            body: JSON.stringify({ error: err.message })
        };
    }
};
