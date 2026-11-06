"""Source-backed dashboard views. Outcome matching is transparent lexical evidence,
not proof of syllabus coverage, learner mastery or commercial whitespace."""
from functools import lru_cache
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import html
import re
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from data import PROJECT, OUTPUT, load_courses
from supplemental import SKILLS

FLAGS={'legacy':'legacy_ai','strict':'strict_ai','expanded':'expanded_ai','student':'student_ai'}
PROPOSALS={
 'LLM':('Evaluate a language-model response against an approved domain reference','Annotated answer audit with unsupported statements flagged'),
 'RAG':('Build a document answering workflow that links answers to approved source passages','Working prototype, citation checks and an out-of-scope question test'),
 'AI agents':('Design a bounded assistant with allowed tools, stop conditions and approval gates','Tool-call trace including a failed-tool recovery and an approval boundary'),
 'Responsible AI':('Evaluate a domain use case for privacy, bias, accountability and escalation','Risk register and a documented review decision'),
 'MLOps':('Deploy and monitor a model with reproducible evaluation and rollback criteria','Versioned deployment artefact and a monitoring incident exercise'),
 'Cloud':('Configure a secure cloud workflow for an AI workload','Reproducible environment with access and cost controls'),
 'Python':('Implement and test a domain-specific data or AI workflow in Python','Runnable notebook or package with tests and a reproducibility note'),
 'SQL':('Query and validate data for a domain-specific AI workflow','SQL queries with data-quality checks and an explanation of assumptions'),
 'Machine learning':('Evaluate a predictive model using a leakage-aware validation design','Baseline comparison and an error analysis on held-out examples'),
 'Generative AI':('Evaluate a generative AI workflow against task-specific acceptance criteria','Test set, grounded outputs and a failure analysis'),
}


def display_time(value):
    return datetime.fromisoformat(value).astimezone(ZoneInfo('Asia/Singapore')).strftime('%d %b %H:%M SGT')


def clean(value):
    if value is None or (not isinstance(value,str) and pd.isna(value)):return ''
    return re.sub(r'\s+',' ',html.unescape(re.sub(r'<[^>]+>',' ',str(value)))).strip()


def usable_outcomes(value):
    text=clean(value)
    section=re.search(r'learning\s+(?:outcomes?|objectives?)\s*:?\s*(.*?)(?=\b(?:topics?\s+covered|course\s+(?:content|outline)|target\s+audience)\s*:|$)',text,re.I)
    if section:text=section.group(1).strip()
    if len(text)<25 or len(text.split())<4:return ''
    if re.fullmatch(r'(?:as per|refer to|see)\s+(?:ctag|website|course outline|course description)\.?',text,re.I):return ''
    return text


def excerpt(text,pattern=None,limit=500):
    text=clean(text)
    match=re.search(pattern,text,re.I) if pattern else None
    start=max(0,match.start()-90) if match else 0
    return ('…' if start else '')+text[start:start+limit]+('…' if len(text)>start+limit else '')


def stamp():
    paths=[OUTPUT/'job_labels.csv',OUTPUT/'course_labels.csv',Path(__file__),Path(__file__).parent/'agentic/policy.json']
    paths.extend(PROJECT/'02_cleaned_data'/n for n in ['mcf_jobs_cleaned_ai_tagged.csv','myskillsfuture_courses_clean.csv','university_courses_clean.csv'])
    return tuple((str(p),p.stat().st_mtime_ns,p.stat().st_size) for p in paths)


@lru_cache(maxsize=4)
def datasets(scenario,version):
    if scenario not in FLAGS:raise ValueError('Unknown evidence scenario')
    jl=pd.read_csv(OUTPUT/'job_labels.csv',usecols=['record_id',FLAGS[scenario],'evidence_label','screening_source','reviewed_by_llm','maturity'])
    jobs=pd.read_csv(PROJECT/'02_cleaned_data/mcf_jobs_cleaned_ai_tagged.csv',usecols=['Job ID','Title','Company','Category','Skills','Description','Job URL','Min Years Exp','Salary Min','Salary Max'],low_memory=False)
    jobs=jobs.merge(jl,left_on='Job ID',right_on='record_id',validate='one_to_one')
    jobs=jobs[jobs[FLAGS[scenario]]].copy()
    jobs['search_text']=jobs.Title.fillna('')+' '+jobs.Skills.fillna('')+' '+jobs.Description.fillna('')
    cet,uni=load_courses()
    uni=uni[(uni.provider!='NUS')|uni.is_currently_offered.eq(True)].copy()
    cet['title']=cet.coursetitle;uni['title']=uni.course_name
    cet['provider_name']=cet.trainingprovideralias;uni['provider_name']=uni.provider
    cet['ecosystem']='CET';uni['ecosystem']='University'
    cet['outcome_field']='what_you_learn';uni['outcome_field']='learning_outcomes'
    cet['source_outcomes']=cet.what_you_learn.map(clean);uni['source_outcomes']=uni.learning_outcomes.map(clean)
    cet['outcomes']=cet.what_you_learn.map(usable_outcomes);uni['outcomes']=uni.learning_outcomes.map(usable_outcomes)
    cet['hours_display']=cet.number_of_hours.map(clean);uni['hours_display']='Not supplied'
    cet['prerequisites']=cet.minimum_entry_requirement.map(clean);uni['prerequisites']=uni.prerequisite.map(clean)
    cols=['record_id','title','provider_name','ecosystem','outcome_field','source_outcomes','outcomes','review_text','hours_display','prerequisites']
    courses=pd.concat([cet[cols],uni[cols]],ignore_index=True)
    cl=pd.read_csv(OUTPUT/'course_labels.csv',usecols=['record_id','legacy_ai','strict_ai','expanded_ai','screening_source'])
    courses=courses.merge(cl,on='record_id',validate='one_to_one')
    course_flag=FLAGS[scenario] if scenario!='student' else 'expanded_ai'
    courses=courses[courses[course_flag]].copy()
    return jobs,courses


def sector_jobs(jobs,sector):
    return jobs[jobs.Category.fillna('').str.split(';').map(lambda xs:sector in [x.strip() for x in xs])].copy()


def retrieve_courses(courses,concept,skills,limit=6):
    """Rank the analysed AI catalogue using declared text similarity; no API calls."""
    if courses.empty:return courses.assign(similarity=pd.Series(dtype=float))
    texts=(courses.title.fillna('')+' '+courses.outcomes.fillna('')+' '+courses.review_text.fillna('')).tolist()
    query=concept+' '+' '.join(skills)
    vectorizer=TfidfVectorizer(stop_words='english',ngram_range=(1,2),max_features=18000,sublinear_tf=True)
    matrix=vectorizer.fit_transform(texts+[query])
    scores=linear_kernel(matrix[-1],matrix[:-1]).ravel()
    selected=np.argsort(-scores,kind='stable')[:limit]
    result=courses.iloc[selected].copy();result['similarity']=scores[selected]
    return result


def coverage_rows(jobs,courses,skills):
    rows=[]
    known=courses.outcomes.ne('')
    for skill in skills:
        pattern=SKILLS[skill]
        demand=jobs.search_text.str.contains(pattern,case=False,regex=True,na=False)
        outcomes=courses.outcomes.str.contains(pattern,case=False,regex=True,na=False)
        descriptions=courses.review_text.str.contains(pattern,case=False,regex=True,na=False)
        source_course=courses.loc[outcomes].iloc[0] if outcomes.any() else None
        quote=source_course.outcomes if source_course is not None else ''
        matching_jobs=jobs.loc[demand]
        source_job=matching_jobs.iloc[0] if not matching_jobs.empty else None
        job='';job_field=''
        for field in ['Description','Skills','Title','search_text']:
            if field not in matching_jobs:continue
            matches=matching_jobs[field].fillna('').str.contains(pattern,case=False,regex=True)
            if matches.any():
                source_job=matching_jobs[matches].iloc[0];job=str(source_job[field]);job_field=field;break
        rows.append({'Skill':skill,'Sector postings mentioning skill':int(demand.sum()),
          'Sector AI postings':len(jobs),'Courses with outcome text':int(known.sum()),
          'Outcome mentions':int(outcomes.sum()),'Description mentions only':int((descriptions&~outcomes).sum()),
          'Outcome text unavailable':int((~known).sum()),
          'Evidence status':'Named in existing outcomes' if outcomes.any() else ('No usable outcome text' if not known.any() else 'Not named in available outcomes'),
          'Existing outcome excerpt':excerpt(quote,pattern),
          'Existing course':str(source_course.get('title','')) if source_course is not None else '',
          'Existing course record':str(source_course.get('record_id','')) if source_course is not None else '',
          'Existing outcome field':str(source_course.get('outcome_field','')) if source_course is not None else '',
          'Employer task excerpt':excerpt(job,pattern),'Employer source field':job_field,
          'Employer source record':str(source_job.get('record_id','')) if source_job is not None else '',
          'Proposed outcome':PROPOSALS.get(skill,(f'Apply {skill} to a domain task and evaluate its limitations','Reproducible task artefact with an error analysis'))[0],
          'Proposed assessment':PROPOSALS.get(skill,('','Reproducible task artefact with an error analysis'))[1]})
    return rows


def next_steps(rows,sector):
    actions=[]
    for r in rows:
        if not r['Sector postings mentioning skill']:
            action='Validate demand before adding this skill'
            why='No matching skill mention in the selected sector AI postings; do not infer demand from the candidate title.'
        elif r['Outcome mentions']:
            action='Extend existing teaching with a domain assessment'
            why=f"{r['Outcome mentions']} of the compared courses name this skill in outcome text. Treat a sector-specific application and assessment as a differentiation hypothesis."
        elif r['Courses with outcome text']:
            action='Investigate an outcome-level addition'
            why='This skill appears in sector postings but is not named in the available outcome text of these retrieved courses. Confirm the syllabi before claiming a gap.'
        else:
            action='Obtain the existing syllabus first'
            why='No usable learning-outcome text is available in the compared catalogue records; coverage is unknown.'
        actions.append({'skill':r['Skill'],'action':action,'why':why,'proposed_outcome':r['Proposed outcome'],
                        'assessment':r['Proposed assessment'],'sector':sector})
    return actions


def outcome_units(text):
    return [p.strip(' .;:-') for p in re.split(r'\bLO\s*\d+\s*[:.\-]?\s*|(?<=[.;])\s+(?=[A-Z])',text) if len(p.strip())>=20]


def outcome_alignment(proposed,compared):
    """Nearest source outcome passage, not whole-course equivalence."""
    units=[(r,unit) for _,r in compared[compared.outcomes.ne('')].iterrows() for unit in outcome_units(r.outcomes)]
    result=[]
    if not proposed:return result
    for outcome in proposed:
        statement=outcome['statement']
        if not units:
            result.append({'Proposed outcome':statement,'Closest source course':'Unknown','Existing outcome text':'Unavailable','Text similarity':None})
            continue
        v=TfidfVectorizer(stop_words='english',ngram_range=(1,2))
        m=v.fit_transform([unit for _,unit in units]+[statement]);sim=linear_kernel(m[-1],m[:-1]).ravel();i=int(np.argmax(sim))
        r,unit=units[i]
        result.append({'Proposed outcome':statement,'Closest source course':r.title,'Existing outcome text':unit,
                       'Source record':r.record_id,'Source field':r.outcome_field,'Text similarity':round(float(sim[i]),3),
                       'Interpretation':'Lexical overlap only' if sim[i]>0 else 'No shared terms after tokenisation'})
    return result
