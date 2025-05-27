#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests
import time
import json
import html
import time
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from transformers import pipeline
import sentencepiece
import pandas as pd
from PIL import Image
import io
import os
import re


# In[ ]:


# Initialize the question generation pipeline
qg_pipeline = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl")


# In[ ]:


def load_and_split_sentences(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        sentences = f.readlines()
    qas = []
    for c, item in enumerate(sentences[0:10000]):
        input_text = f"generate question: {item}"
        generated_question = qg_pipeline(input_text, max_length=512, do_sample=False)[0]['generated_text']
        print(f'{c} -> /n {generated_question}/n {item}')
        yield {
            'question': generated_question,
            'answer': item,
            'source': 'SpaceSystemsDataset'
        }
        
    # return qas
    


# In[ ]:


# space_systems_qas = list(load_and_split_sentences('/Users/rckyi/Documents/Data/SpaceSystemsDataset/2 SpaceTransformersCorpus/Sentences_WikiBooksAbstracts.txt'))


# In[ ]:


# len(space_systems_qas)


# In[ ]:


def load_and_split_paragraphs(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split on one or more blank lines (handles multiple \n between paragraphs)
    import re
    pattern = r"(*Figure -*\s*\s*-\s*(\d+)\s*"
    pat = r"[ ]*(\d+[ ]*)|\(*Figure -*\s*\d+\)?"
    # paragraphs = [para.strip() for para in re.split(r'\n\s*\n', re.sub(pattern, " ", text)) if para.strip()]
    paragraphs = [para.strip().replace("- . ","").replace(" . ","") for para in re.split(r'\n\s*\n', re.sub(r"\s*\d+\s*", "", re.sub(pat, " ", text))) if para.strip()]
    
    return paragraphs


# In[ ]:


# paragraphs = load_and_split_paragraphs("/Users/rckyi/Documents/Data/Nasa-Lessons-learned-in-engineering.txt")

# lessons_learned_qa = []
# for i, para in enumerate(paragraphs):  # Print first 5 paragraph
#     input_text = f"generate question: {para}"
#     generated_question = qg_pipeline(input_text, max_length=512, do_sample=False)[0]['generated_text']
#     lessons_learned_qa.append({
#             'question': generated_question,
#             'answer': para,
#             'source': 'nasa: lessons learned'
#     })
    
    # print(f"\n--- Paragraph {i + 1} ---\n{generated_question}\n{para}")
        


# In[ ]:


# paragraphs_ps = split_into_paragraphs("/Users/rckyi/Documents/Data/A HISTORY OF AEROSPACE PROBLEMS, THEIR SOLUTIONS, THEIR LESSONS")


# In[ ]:


# for i, para in enumerate(paragraphs_ps[200:300]):  # Print first 5 paragraph
#     input_text = f"generate question: {para}"
#     generated_question = qg_pipeline(input_text, max_length=512, do_sample=False)[0]['generated_text']
#     print(f"\n--- Paragraph {i + 1} ---\n{generated_question}\n{para}")


# In[ ]:


def process_paragraphs(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        paragraphs = json.load(f)

    results = []
    for para_id, para_text in paragraphs.items():
        input_text = f"generate question: {para_text}"
        results.append({'question': qg_pipeline(input_text, max_length=712, do_sample=False)[0]['generated_text'], 
                        'answer': para_text,
                       'source':  f'nasa: a history of aerospace problems and solns'
                       })
        # yield [qg_pipeline(input_text, max_length=512, do_sample=False)[0]['generated_text'], para_text]

    return results


# In[ ]:


# Example usage:
json_file_path = "/Users/rckyi/Documents/Data/paragraphs_with_ids.json"  # or your full path


# In[ ]:


# arXiv API Q&A extraction from abstract (pseudo-QA from title and abstract)
def fetch_arxiv_abstracts(query='rocket propulsion', max_results=100):
    url = f"http://export.arxiv.org/api/query?search_query=all:{query.replace(' ', '+')}&start=0&max_results={max_results}"
    resp = requests.get(url)
    root = ET.fromstring(resp.content)

    qas = []
    ns = {'atom': 'http://www.w3.org/2005/Atom'}  # arXiv uses Atom XML namespace

    for entry in root.findall('atom:entry', ns):
        title = entry.find('atom:title', ns).text.strip()
        
        summary = entry.find('atom:summary', ns).text.strip()
        # Generate question from text
        input_text = f"generate question: {summary}"
        generated_question = qg_pipeline(input_text, max_length=64, do_sample=False)[0]['generated_text']
        
        qas.append({
            # 'title': title,
            'question': generated_question,
            'answer': summary,
            'source': f'arxiv'
        })
    return qas


# In[ ]:


# Wikipedia scraping

# Initialize the question generation pipeline
# qg_pipeline = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl")

def clean_text(text):
    return ' '.join(text.strip().split())

def get_all_rocket_propulsion_links(base_url='https://en.wikibooks.org/wiki/Rocket_Propulsion'):
    """Grab all unique subpage links under Rocket Propulsion."""
    resp = requests.get(base_url)
    soup = BeautifulSoup(resp.text, 'html.parser')

    content_div = soup.select_one('#mw-content-text')
    links = content_div.find_all('a', href=True)
    
    urls = set()
    for link in links:
        href = link['href']
        if 'https' in href:
            full_url = href
        else:
            full_url = 'https://en.wikibooks.org' + href
        urls.add(full_url)

    return list(urls)

def scrape_pages_with_qg(urls, visited, M, batch_size=10):
    """Scrape up to `batch_size` new pages not in `visited`, return new QAs."""
    qa_pairs = []
    count = 0

    for url in urls:
        if url in visited:
            continue
        visited.add(url)
        try:
            resp = requests.get(url)
            soup = BeautifulSoup(resp.text, 'html.parser')
            content_div = soup.select_one('#mw-content-text')
            paragraphs = content_div.find_all('p')
            text = clean_text(' '.join(p.get_text() for p in paragraphs[:3]))
            if not text or len(text) < M:
                continue

            # Generate question from text
            input_text = f"generate question: {text}"
            output = qg_pipeline(input_text, max_length=64, do_sample=False)[0]['generated_text']

            qa_pairs.append({
                'question': output,
                'answer': text,
                'source': f"rocketry wiki"
            })
            print(f"✅ Generated Q&A from: {url}")
            count += 1
            time.sleep(1)  # polite scraping

            if count >= batch_size:
                break
        except Exception as e:
            print(f"⚠️ Failed to process {url}: {e}")
    return qa_pairs

def scrape_wikibook_qas(M):
    all_links = get_all_rocket_propulsion_links()
    visited = set()
    all_qas = []

    for i in range(10):
        print(f"\n🔁 Batch {i+1}/10")
        batch_qas = scrape_pages_with_qg(all_links, visited, M, batch_size=10)
        all_qas.extend(batch_qas)
        if len(all_qas) >= M:
            break

    return all_qas


# In[ ]:


# Stack Exchange API

def clean_html_text(html_content):
    """Remove hyperlinks and strip HTML tags from content."""
    soup = BeautifulSoup(html_content, 'html.parser')

    # Replace <a> tags with their inner text
    for a in soup.find_all('a'):
        a.replace_with(a.get_text())

    # Get cleaned text
    text = soup.get_text(separator=' ')
    return html.unescape(text.strip())

def fetch_stackexchange_qas(site='space.stackexchange', tag='rockets', pagesize=20, max_pages=10):
    base_url = 'https://api.stackexchange.com/2.3/questions'
    answers_url = 'https://api.stackexchange.com/2.3/questions/{ids}/answers'
    all_qas = []

    for page in range(1, max_pages + 1):
        params = {
            'site': site,
            'tagged': tag,
            'pagesize': pagesize,
            'page': page,
            'filter': 'withbody'
        }
        resp = requests.get(base_url, params=params)
        print(resp)
        data = resp.json()

        for question in data.get('items', []):
            q_id = question['question_id']
            q_body = clean_html_text(question.get('body', ''))
            title = html.unescape(question.get('title', ''))

            # Get answers
            a_params = {
                'site': site,
                'filter': 'withbody'
            }
            a_resp = requests.get(answers_url.format(ids=q_id), params=a_params)
            answers = a_resp.json().get('items', [])

            for ans in answers:
                a_body = clean_html_text(ans.get('body', ''))
                all_qas.append({
                    'question': f"{title}\n{q_body}",
                    'answer': a_body,
                    'source': f"https://{site}.com/questions/{q_id}"
                })

            time.sleep(20)  # polite API usage

    return all_qas


# In[ ]:


# se_data = fetch_stackexchange_qas()


# In[ ]:


if __name__ == "__main__":
    
    # arxiv_data = fetch_arxiv_abstracts()
    # reddit_data = fetch_reddit_qa()
    # wikibook_data = scrape_wikibook_qas(M=1000)
    space_systems_qas = list(load_and_split_sentences('/Users/rckyi/Documents/Data/SpaceSystemsDataset/2 SpaceTransformersCorpus/Sentences_WikiBooksAbstracts.txt'))
    # nasa_probs_solns = process_paragraphs(json_file_path)

    all_qas = {
        # "stackexchange": se_data,
        # "arxiv": arxiv_data,
        # "reddit": reddit_data,
        # "wikibook": wikibook_data,
        "spacesystems": space_systems_qas
        # "nasa lessons learned": lessons_learned_qa
        # "nasa problems and solns": nasa_probs_solns
    }

    with open('/Users/rckyi/Documents/Data/space_systems_qas_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(all_qas, f, indent=2)

    print(f"\n✅ Done. Saved {len(all_qas)} Q&A pairs to qas_data.json")


# In[ ]:




