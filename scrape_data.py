print("Importing necessary libraries...")
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import os

print("Libraries imported successfully.")

def scrape_shl_catalog_with_pdfs():
    base_url = "https://www.shl.com/solutions/products/product-catalog/"
    assessments = []
    
    # Ensure the data directory exists
    print("Creating directory 'data/pdfs' if it doesn't exist...")
    os.makedirs("data/pdfs", exist_ok=True)
    print("Directory setup complete.")
    
    print("Starting SHL catalog scraping...")
    for page in range(0, 32 * 12, 12):
        url = f"{base_url}?start={page}&type=2&type=1"
        print(f"\nScraping page at offset {page}: {url}")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        print(f"Page response status code: {response.status_code}")
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Get the first (and likely only) custom__table-wrapper div
        table_divs = soup.find_all('div', class_='custom__table-wrapper')
        if not table_divs:
            print(f"No table found on page {page}. Skipping...")
            continue
        print(f"Found {len(table_divs)} table(s) on page {page}")
        
        table = table_divs[0]  # Use the first div
        rows = table.find_all('tr')[1:]  # Skip the header row
        print(f"Processing {len(rows)} assessments on page {page}")
        
        for row in rows:
            cols = row.find_all('td')
            name_link = cols[0].find('a')
            name = name_link.text.strip()
            assessment_url = "https://www.shl.com" + name_link['href']
            remote = "Yes" if cols[1].find('span', class_='catalogue__circle -yes') else "No"
            adaptive = "Yes" if cols[2].find('span', class_='catalogue__circle -yes') else "No"
            test_type = cols[3].text.strip() or "N/A"
            
            print(f"\nProcessing assessment: {name}")
            print(f"Assessment URL: {assessment_url}")
            print(f"Remote: {remote}, Adaptive: {adaptive}, Test Type: {test_type}")
            
            # Fetch the assessment page
            assessment_response = requests.get(assessment_url)
            print(f"Assessment page status code: {assessment_response.status_code}")
            assessment_soup = BeautifulSoup(assessment_response.text, 'html.parser')
            
            # Extract data from the assessment page
            description = ""
            job_levels = ""
            languages = ""
            assessment_length = "N/A"
            
            print("Extracting data from assessment page...")
            sections = assessment_soup.find_all('div', class_='product-catalogue-training-calendar__row')
            for section in sections:
                h4 = section.find('h4')
                if h4:
                    title = h4.text.strip().lower()
                    p = section.find('p')
                    text = p.text.strip() if p else ""
                    
                    if "description" in title:
                        description = text
                        print(f"Description: {description}")
                    elif "job levels" in title:
                        job_levels = text
                        print(f"Job Levels: {job_levels}")
                    elif "languages" in title:
                        languages = text
                        print(f"Languages: {languages}")
                    elif "assessment length" in title:
                        assessment_length_match = re.search(r"(\d+\s*minutes)", text, re.IGNORECASE)
                        assessment_length = assessment_length_match.group(1) if assessment_length_match else text
                        print(f"Assessment Length: {assessment_length}")
            
            # Download the PDF
            pdf_link_elem = assessment_soup.find('p', class_='product-catalogue__download-title')
            pdf_url = pdf_link_elem.find('a')['href'] if pdf_link_elem and pdf_link_elem.find('a') else None
            if pdf_url:
                print(f"Found PDF URL: {pdf_url}")
                try:
                    pdf_response = requests.get(pdf_url)
                    print(f"PDF response status code: {pdf_response.status_code}")
                    # Create a safe filename by replacing invalid characters
                    safe_name = re.sub(r'[<>:"/\\|?*]', '_', name)
                    pdf_path = f"data/pdfs/{safe_name}.pdf"
                    with open(pdf_path, "wb") as f:
                        f.write(pdf_response.content)
                    print(f"Downloaded PDF for {name} to {pdf_path}")
                except Exception as e:
                    print(f"Error downloading PDF for {name}: {e}")
            else:
                print(f"No PDF URL found for {name}")
            
            # Append the extracted data
            assessments.append({
                "name": name,
                "url": assessment_url,
                "remote": remote,
                "adaptive": adaptive,
                "test_type": test_type,
                "description": description,
                "job_levels": job_levels,
                "languages": languages,
                "assessment_length": assessment_length
            })
            print(f"Added {name} to assessments list")
            time.sleep(1)  # Be polite to the server
        
    print(f"\nTotal assessments scraped: {len(assessments)}")
    df = pd.DataFrame(assessments)
    df.to_csv("data/shl_assessments_with_pdfs.csv", index=False)
    print("Data saved to 'data/shl_assessments_with_pdfs.csv'")
    return df

if __name__ == "__main__":
    scrape_shl_catalog_with_pdfs()
    print("Data scraping complete.")