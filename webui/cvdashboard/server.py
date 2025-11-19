#!/usr/bin/env python3
"""
Simple server for the Job Dashboard that serves CSV data as JSON.
Handles large CSV files efficiently.
"""

import csv
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import os
from pathlib import Path

CSV_PATH = Path(__file__).parent.parent.parent / 'data' / 'job_postings.csv'
CACHE = {'data': None, 'dropdowns': None}

def load_csv_sample(limit=None):
    """Load CSV with optional row limit."""
    data = []
    try:
        with open(CSV_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if limit and i >= limit:
                    break
                data.append(row)
    except Exception as e:
        print(f"Error loading CSV: {e}")
    return data

def get_dropdowns():
    """Extract unique values for dropdown filters."""
    if CACHE['dropdowns']:
        return CACHE['dropdowns']
    
    data = load_csv_sample()  # Load all for unique extraction
    
    companies = sorted(set(r.get('company_name', '').strip() for r in data if r.get('company_name', '').strip()))
    titles = sorted(set(r.get('job_title', '').strip() for r in data if r.get('job_title', '').strip()))
    locations = sorted(set(r.get('job_location', '').strip() for r in data if r.get('job_location', '').strip()))
    industries = sorted(set(r.get('job_industries', '').strip() for r in data if r.get('job_industries', '').strip()))
    
    result = {
        'companies': companies,
        'titles': titles,
        'locations': locations,
        'industries': industries
    }
    CACHE['dropdowns'] = result
    return result

def filter_jobs(company='', title='', location='', industry='', min_salary=0, max_salary=None, limit=100):
    """Filter jobs based on criteria."""
    data = load_csv_sample()
    filtered = []
    
    for row in data:
        if company and row.get('company_name', '').strip() != company:
            continue
        if title and row.get('job_title', '').strip() != title:
            continue
        if location and row.get('job_location', '').strip() != location:
            continue
        if industry and row.get('job_industries', '').strip() != industry:
            continue
        
        # Handle salary filtering
        salary_str = row.get('job_base_pay_range', '').strip()
        if salary_str and (min_salary or max_salary):
            try:
                # Extract first number from salary string (e.g., "$100,000.00/yr - $120,000.00/yr")
                import re
                numbers = re.findall(r'[\d,]+\.?\d*', salary_str)
                if numbers:
                    salary_val = float(numbers[0].replace(',', ''))
                    if min_salary and salary_val < min_salary:
                        continue
                    if max_salary and salary_val > max_salary:
                        continue
            except:
                pass  # Skip if can't parse
        
        filtered.append(row)
        if len(filtered) >= limit:
            break
    
    return filtered

class JobDashboardHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        
        # API endpoint for dropdowns
        if parsed.path == '/api/dropdowns':
            try:
                dropdowns = get_dropdowns()
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(dropdowns).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
            return
        
        # API endpoint for filtered jobs
        if parsed.path == '/api/jobs':
            try:
                query = parse_qs(parsed.query)
                company = query.get('company', [''])[0]
                title = query.get('title', [''])[0]
                location = query.get('location', [''])[0]
                industry = query.get('industry', [''])[0]
                min_salary = int(query.get('minSalary', ['0'])[0]) if query.get('minSalary', ['0'])[0] else 0
                max_salary = int(query.get('maxSalary', ['999999999'])[0]) if query.get('maxSalary') else None
                search = query.get('search', [''])[0]
                limit = int(query.get('limit', ['100'])[0])
                
                jobs = filter_jobs(company, title, location, industry, min_salary, max_salary, limit)
                
                # Apply global search filter on client data
                if search:
                    search_lower = search.lower()
                    jobs = [j for j in jobs if any(
                        search_lower in str(j.get(field, '')).lower()
                        for field in ['job_title', 'company_name', 'job_location', 'job_industries', 'job_summary']
                    )]
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(jobs).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
            return
        
        # Serve static files
        super().do_GET()
    
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        self.send_header('Expires', '0')
        super().end_headers()

if __name__ == '__main__':
    os.chdir(Path(__file__).parent)
    server = HTTPServer(('localhost', 8000), JobDashboardHandler)
    print('Server running at http://localhost:8000')
    print(f'CSV file: {CSV_PATH}')
    server.serve_forever()
