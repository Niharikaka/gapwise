"""All provider statistics and rankings are computed here, never by an LLM.
Course-to-sector membership is explicitly a text proxy over the AI catalogue.
"""
from pathlib import Path
from functools import lru_cache
import hashlib
import json
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import linear_kernel
from data import PROJECT,OUTPUT
from supplemental import SKILLS
from dashboard_data import datasets,stamp,sector_jobs,retrieve_courses,outcome_units
from agentic.evidence import ALIASES,sha
from .brief import ProviderBrief

HERE=Path(__file__).resolve().parent
DEST=OUTPUT/'provider_demo'
POLICY=json.loads((HERE/'policy.json').read_text())
CATALOGUE=[
 {'key':'assurance','name':'AI assurance and governance review','skills':['Responsible AI','Generative AI','LLM'],'levels':[1,2],
  'intent':'Managers consultants risk compliance officers review AI use cases governance policy privacy bias controls accountability approval responsible business adoption'},
 {'key':'productivity','name':'Generative AI for everyday business work','skills':['Generative AI','LLM','AI agents'],'levels':[1,2],
  'intent':'Non-technical staff executives administrators sales marketing HR teams use generative AI assistants copilots to draft summarise write reports emails proposals presentations content prompts review outputs everyday productivity'},
 {'key':'documents','name':'Evidence-checked document answering','skills':['LLM','RAG','Responsible AI'],'levels':[2,3],
  'intent':'Professionals answer questions over approved internal documents policies manuals procedures retrieval sources citations evidence checks document question answering knowledge assistant search'},
 {'key':'language','name':'Customer and text intelligence with NLP','skills':['NLP','LLM','Generative AI'],'levels':[2,3],
  'intent':'Customer service marketing HR legal teams classify analyse text customer enquiries feedback sentiment complaints chatbots contracts resumes emails automate responses'},
 {'key':'analytics','name':'AI-assisted data analysis and forecasting','skills':['Machine learning','Python','SQL'],'levels':[2,3],
  'intent':'Analysts business operations teams clean analyse data dashboards reporting forecast sales demand trends customer segmentation insights spreadsheets decision making'},
 {'key':'workflow','name':'Bounded AI workflow and tool integration','skills':['AI agents','LLM','RAG'],'levels':[3],
  'intent':'Software engineers technical teams integrate AI agents tools document retrieval assistants into internal workflows automation stop conditions approval recover failed calls'},
 {'key':'prediction','name':'Predictive decision support and model evaluation','skills':['Machine learning','Python','Data engineering'],'levels':[3,4],
  'intent':'Engineers analysts operations teams predictive maintenance risk scoring evaluate predictive models data validation failure detection operational constraints baseline error analysis'},
 {'key':'vision','name':'Computer vision for inspection and monitoring','skills':['Computer vision','Deep learning','Python'],'levels':[3,4],
  'intent':'Engineers technicians healthcare logistics retail teams image video inspection defect detection quality control medical imaging safety monitoring object detection cameras'},
 {'key':'production','name':'Production AI reliability and monitoring','skills':['MLOps','Cloud','Python'],'levels':[4,5],
  'intent':'ML software engineers deploy operate monitor production AI models cloud model serving incident recovery rollback reliability drift performance'}
]
# A course counts toward an industry only when its title names that industry. Matching the
# full description over-counts incidental mentions ("Classroom", "Programme", "prompt engineering").
# Revised after a blind stratified review (outputs/provider_demo/sector_mapping_validation): bare generic
# words ("computing", "executives", "sustainable", "user interface") were dropped, and common ways of
# naming a sector without its category word ("law", "leading", "Azure", "facade") were added.
SECTOR_PATTERNS={
 'Banking and Finance':r'\b(?:financ\w*|bank\w*|treasury|fintech|wealth|investments?|investors?|capital markets?|valuation|credit (?:risk|scoring|analysis)|trading|payments?|anti-money laundering)\b',
 'Information Technology':r'\b(?:software|programmers?|developers?|information technology|computer science|data engineer\w*|cloud|devops|mlops|cyber\w*|azure|aws|amazon web services|google cloud|full[- ]stack|web (?:development|developers?)|app(?:lication)? development|it (?:professionals|networking|operations|infrastructure|security)|network security|security tools|server administration)\b',
 'Engineering':r'\b(?:mechanical|electrical|electronics?|civil|chemical engineering|aerospace|marine|automotive|robots?|robotics|predictive maintenance|maintenance engineer\w*|signal processing|machine vision|power systems?|autonomous (?:vehicles?|driving)|engineers|diploma in engineering|engineering (?:design|operations|teams?|professionals|firms?|industry|sector|applications))\b',
 'Manufacturing':r'\b(?:manufactur\w*|factory|factories|shop ?floor|production lines?|industrial|semiconductor\w*|precision engineering|industry [45]\.0|quality inspection)\b',
 'Consulting':r'\b(?:(?:business|management|technology|strategy|digital transformation) consult\w*|consulting (?:firms?|practice|skills)|advisory (?:firms?|services))\b',
 'Marketing / Public Relations':r'\b(?:marketing|marketers?|advertis\w*|campaigns?|public relations|brand\w*|social media|content (?:marketing|strategy|commerce)|seo)\b',
 'Sciences / Laboratory / R&D':r'\b(?:laborator\w*|life sciences?|biolog\w*|chemistry|biotech\w*|research and development|r&d|scientists?|drug discovery|genomic\w*|bioinformatics|neuroscien\w*|optics|photonics|physics|materials (?:science|ai|discovery)|therapeutics)\b',
 'Sales / Retail':r'\b(?:sales|retail\w*|e-?commerce|merchandis\w*|salespersons?|shoppers?|selling|commercial excellence|account management)\b',
 'Admin / Secretarial':r'\b(?:admin(?:istrative)? (?:staff|support|assistants?|work|tasks|executives?|professionals)|office administrat\w*|secretar\w*|clerical|office (?:productivity|tasks)|executive assistants?|personal assistants?)\b',
 'Accounting / Auditing / Taxation':r'\b(?:accounting|accountan\w*|auditing|auditors?|financial audits?|tax(?:ation|es)?|bookkeep\w*|financial reporting|financial statements?|(?<!quality )assurance)\b',
 'Customer Service':r'\b(?:customer (?:service|support|experience|care|enquir\w*|queries)|call cent(?:re|er)s?|contact cent(?:re|er)s?|helpdesk|service desk|service excellence|service leadership|guest experience)\b',
 'Advertising / Media':r'\b(?:advertis\w*|(?<!social )(?<!and digital )media\b|journalis\w*|news|broadcast\w*|publishing|content creators?|(?:digital|video|creative) content|video (?:production|creation)|script ?writing|(?<!data )storytelling)\b',
 'Human Resources':r'\b(?:human resources?|hr|recruit\w*|talent (?:acquisition|management)|hiring|people management|workforce (?:planning|learning|development)|payroll)\b',
 'Professional Services':r'\b(?:professional services|professional firms?|advisory (?:firms?|services)|client services)\b',
 'Education and Training':r'\b(?:educators?|educational|edtech|teachers?|teaching|pedagog\w*|curriculum|learning design|instructional design|schools?|lecturers?|trainers?|for training|learning and development)\b',
 'Design':r'\b(?:graphic design\w*|designers?|ux|ui/ux|user experience|visual design|interior design|product design|creative professionals)\b',
 'Logistics / Supply Chain':r'\b(?:logistic\w*|supply chains?|warehous\w*|procurement|inventory|freight|shipping|fleet)\b',
 'Healthcare / Pharmaceutical':r'\b(?:healthcare|health care|hospitals?|medical|medicine|clinical|clinicians?|patients?|nurses?|nursing|pharma\w*|digital health|health informatics)\b',
 'General Management':r'\b(?:general management|senior management|business (?:leaders?|owners?|strategy)|leaders?|leading|leadership|managerial|c-suite|chief \w+ officers?|entrepreneur\w*|managers|innovation management|strategic (?:innovation|management))\b',
 'Environment / Health':r'\b(?:environmental|environment(?:al)? sustainability|sustainab\w* (?:technolog|manufactur|chemistr|energy|operations|building|development goals)\w*|green (?:technolog\w*|buildings?|vehicles?|energy)|climate|carbon|esg|workplace safety|health and safety|occupational (?:health|safety)|waste management|energy efficiency|hydro\w*)\b',
 'Insurance':r'\b(?:insur\w*|underwrit\w*|actuar\w*|claims processing|reinsurance)\b',
 'Events / Promotions':r'\b(?:events? (?:management|planning|planners?|industry)|event organis\w*|exhibitions?|trade shows?|mice)\b',
 'Risk Management':r'\b(?:risk management|risk managers?|risk assessments?|enterprise risk|credit risk|operational risk|financial risk|governance,? risk|data protection|compliance|financial crime)\b',
 'Wholesale Trade':r'\b(?:wholesal\w*|distributors?|trading compan\w*)\b',
 'Medical / Therapy Services':r'\b(?:therapists?|therapy|physiotherap\w*|occupational therap\w*|rehabilitat\w*|allied health|clinics?)\b',
 'Telecommunications':r'\b(?:telecom\w*|telcos?|network operators?|5g|o-?ran|open ran|radio access networks?|mobile networks?)\b',
 'Legal':r'\b(?:legal|laws?|lawyers?|law firms?|litigation|paralegal\w*|justice)\b',
 'Building and Construction':r'\b(?:construction|built environment|building information model\w*|bim|quantity survey\w*|facilit(?:y|ies) management|civil engineering|fa[cç]ades?|green buildings?|smart buildings?|building services|real estate)\b',
}
SUPPORTED_SECTORS=sorted(SECTOR_PATTERNS)
GENERIC_AUDIENCE={'working','professional','professionals','mid','career','experienced','experience','basic','skills','skill','with','and','the','for','in','of','to','a','an','who','people','learner','learners','team','teams'}


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def source_hashes(scenario):
    paths=[PROJECT/'02_cleaned_data'/n for n in ['mcf_jobs_cleaned_ai_tagged.csv','myskillsfuture_courses_clean.csv','university_courses_clean.csv']]
    paths += [OUTPUT/'job_labels.csv',OUTPUT/'course_labels.csv',OUTPUT/'metrics'/scenario/'tables/sec.csv',HERE/'policy.json',HERE/'research.py',HERE/'brief.py',Path(__file__).parent.parent/'dashboard_data.py',Path(__file__).parent.parent/'data.py']
    return {str(p):sha(p) for p in paths}


def canonical(sector,sectors):
    # An exact category wins over an alias, so "Manufacturing" is not folded into Engineering.
    for requested in [sector.strip(),ALIASES.get(sector.strip().lower(),sector.strip())]:
        match=next((s for s in sectors if s.casefold()==requested.casefold()),None)
        if match:return match
    return None


def sector_pattern(sector):
    # Unmapped categories (e.g. "Others") get no supply count rather than a guessed keyword.
    return SECTOR_PATTERNS.get(sector)


def audience_tokens(audience):
    tokens=[t for t in re.findall(r'[a-z]{3,}',audience.lower()) if t not in GENERIC_AUDIENCE]
    return sorted(set(re.sub(r'(?:ing|s)$','',t) if len(t)>5 else t for t in tokens))


SUFFIXES=('ations','ation','ives','ive','ings','ing','ions','ion','ers','er','ies','es','ed','s')


def stem_tokens(text):
    # Light suffix stripping so "defects"/"detection" meet "defect detection" in brief fit.
    tokens=[t for t in re.findall(r'[a-z0-9]+',text.lower()) if t not in ENGLISH_STOP_WORDS]
    return [next((t[:-len(s)] for s in SUFFIXES if t.endswith(s) and len(t)-len(s)>=4),t) for t in tokens]


def similarities(query,texts):
    if not texts:return np.array([])
    v=TfidfVectorizer(tokenizer=stem_tokens,token_pattern=None,lowercase=False,ngram_range=(1,2),sublinear_tf=True)
    m=v.fit_transform(texts+[query])
    return linear_kernel(m[-1],m[:-1]).ravel()


def facts_add(facts,key,value,unit,statement,source,row):
    identifier='fact_'+digest([key,str(source),row])[:12]
    facts[identifier]={'id':identifier,'key':key,'value':value,'unit':unit,'statement':statement,
                       'source':{'file':str(source),'sha256':sha(source),'row':row}}
    return identifier


def course_evidence(frame):
    result=[]
    for _,r in frame.iterrows():
        units=outcome_units(r.outcomes)
        result.append({'course_id':r.record_id,'title':r.title,'provider':r.provider_name,'ecosystem':r.ecosystem,
           'outcome_field':r.outcome_field,'outcomes_available':bool(units),
           'outcomes':[{'outcome_id':'outcome_'+digest([r.record_id,i,text])[:12],'quote':text[:700],'field':r.outcome_field} for i,text in enumerate(units[:4])],
           'source_excerpt':r.review_text[:400],'prerequisites':r.prerequisites,'hours':r.hours_display,
           'retrieval_similarity':float(r.similarity)})
    return result


def rule_results(c,policy=POLICY):
    return [
     {'rule':'enough_demand','label':'Observed hiring signal','passed':c['sector_ai_jobs']>=policy['minimum_sector_ai_jobs'] and c['candidate_skill_jobs']>=policy['minimum_candidate_skill_jobs'] and c['explicit_support_share']>=policy['minimum_explicit_support_share'],
      'observed':{'sector_ai_jobs':c['sector_ai_jobs'],'candidate_skill_jobs':c['candidate_skill_jobs'],'explicit_support_share':c['explicit_support_share']},
      'threshold':{'sector_jobs':policy['minimum_sector_ai_jobs'],'candidate_jobs':policy['minimum_candidate_skill_jobs'],'explicit_support':policy['minimum_explicit_support_share']}},
     {'rule':'limited_supply','label':'Supply strategy: adapt existing offering' if c.get('adaptation_route') else 'Sector course / job ratio','passed':c['sector_courses_per_100_ai_jobs'] is not None and (c['sector_courses_per_100_ai_jobs']<=policy['maximum_sector_courses_per_100_ai_jobs'] or c.get('adaptation_route',False)),
      'observed':{'sector_course_entries':c['sector_course_entries'],'sector_ai_jobs':c['sector_ai_jobs'],'courses_per_100_ai_jobs':c['sector_courses_per_100_ai_jobs'],'route':'adapt existing' if c.get('adaptation_route') else 'new launch screening','new_launch_supply_pass':c['sector_courses_per_100_ai_jobs'] is not None and c['sector_courses_per_100_ai_jobs']<=policy['maximum_sector_courses_per_100_ai_jobs']},'threshold':policy['maximum_sector_courses_per_100_ai_jobs']},
     {'rule':'clear_audience','label':'Audience connects to real job titles','passed':c['audience_title_matches']>=policy['minimum_audience_title_matches'],
      'observed':{'title_matches':c['audience_title_matches'],'screened_terms':c['audience_terms']},'threshold':policy['minimum_audience_title_matches']},
     {'rule':'right_depth','label':'Depth matches the brief and sector','passed':bool(c['overlap_levels']) and c['sector_depth_share']>=policy['minimum_sector_depth_share'],
      'observed':{'candidate_levels':c['levels'],'overlap':c['overlap_levels'],'sector_depth_share':c['sector_depth_share']},'threshold':policy['minimum_sector_depth_share']},
     {'rule':'skill_gap','label':'Measured skill-mention gap','passed':c['mean_positive_skill_gap']>=policy['minimum_mean_positive_skill_gap'],
      'observed':c['mean_positive_skill_gap'],'threshold':policy['minimum_mean_positive_skill_gap']},
     {'rule':'differentiation','label':'Brief fit and actual outcome evidence','passed':c['brief_similarity']>=policy['minimum_brief_similarity'] and c['known_outcome_courses']>=policy['minimum_known_outcome_courses'] and c.get('baseline_outcomes_ok',True),
      'observed':{'brief_similarity':c['brief_similarity'],'known_outcome_courses':c['known_outcome_courses']},'threshold':{'brief_similarity':policy['minimum_brief_similarity'],'known_courses':policy['minimum_known_outcome_courses']}}
    ]


def sensitivity(candidates,seed=588):
    rng=np.random.default_rng(seed);keys=list(POLICY['weights']);weights=np.array(list(POLICY['weights'].values()))
    eligible=[c for c in candidates if c['eligible']]
    wins={c['candidate_id']:0 for c in candidates}
    if eligible:
        draws=rng.dirichlet(weights*20,size=200)
        values=np.array([[c['components'][k] for k in keys] for c in eligible])
        for w in draws:wins[eligible[int(np.argmax(values@w))]['candidate_id']]+=1
    threshold_results={c['candidate_id']:0 for c in candidates}
    grid=[]
    for demand in [10,25,50]:
      for ratio in [50,100,150]:
       for gap in [0.0,0.005,0.01]:
        p=dict(POLICY,minimum_sector_ai_jobs=demand,maximum_sector_courses_per_100_ai_jobs=ratio,minimum_mean_positive_skill_gap=gap)
        ids=[c['candidate_id'] for c in candidates if all(r['passed'] for r in rule_results(c,p))]
        for cid in ids:threshold_results[cid]+=1
        grid.append({'min_sector_jobs':demand,'max_courses_per_100_jobs':ratio,'min_gap':gap,'eligible':ids})
    return {'weight_draws':200,'dirichlet_concentration':20,'random_seed':seed,
            'weight_top_share':{cid:n/200 for cid,n in wins.items()},
            'threshold_scenarios':27,'threshold_eligible_share':{cid:n/27 for cid,n in threshold_results.items()},
            'threshold_grid':grid,'note':'Heuristic weight/threshold stress test, not a confidence interval or probability of commercial success.'}


@lru_cache(maxsize=20)
def _analyse(serialised,version):
    brief=ProviderBrief(**json.loads(serialised)).validate()
    base=OUTPUT/'metrics'/brief.scenario
    manifest=json.loads((base/'run_manifest.json').read_text())
    if manifest['status']!='complete' or manifest['job_labels_sha256']!=sha(OUTPUT/'job_labels.csv') or manifest['course_labels_sha256']!=sha(OUTPUT/'course_labels.csv'):
        raise ValueError('Rebuild the analytical evidence after source-label changes')
    sectors=pd.read_csv(base/'tables/sec.csv');sector=canonical(brief.sector,sectors.sector.tolist())
    bundle={'version':POLICY['version'],'brief':brief.to_dict(),'sector':sector,'policy':POLICY,'source_hashes':source_hashes(brief.scenario),
            'facts':{},'candidates':[],'recommended_candidate_id':None,'status':'insufficient_evidence',
            'interpretation':['Sector job counts use exact category membership; courses count toward a sector only when their title names it.',
              'Skill gaps compare sector AI-posting text with the pooled AI-course catalogue; they are mention-share proxies.',
              'Unknown outcome text is never scored as absent teaching. Outcome retrieval is a shortlist, not the full market.',
              'Candidate options and statistical measures are decision support; substantive fit and novelty require provider validation.']}
    if sector is None:return bundle
    jobs,courses=datasets(brief.scenario,stamp());j=sector_jobs(jobs,sector)
    if brief.baseline_course_id and brief.baseline_course_id not in set(courses.record_id):raise ValueError('The selected baseline course is outside this scenario’s analysed AI catalogue')
    sr=sectors.set_index('sector').loc[sector]
    cpat=sector_pattern(sector)
    sector_courses=courses[courses.title.str.contains(cpat,case=False,regex=True,na=False)] if cpat else courses.iloc[:0]
    ratio=100*len(sector_courses)/len(j) if len(j) and cpat else None
    support=float(j.evidence_label.eq('explicit').mean()) if len(j) else 0
    terms=audience_tokens(brief.audience)
    amask=j.Title.str.contains('|'.join(re.escape(t) for t in terms),case=False,regex=True,na=False) if terms else pd.Series(False,index=j.index)
    fit=similarities(brief.audience+' '+brief.business_need,[c['intent'] for c in CATALOGUE])
    facts=bundle['facts'];file=base/'tables/sec.csv';row={'sector':sector}
    market_ids=[facts_add(facts,'sector_ai_jobs',len(j),'postings',f'AI-posting count in {sector}: {len(j)}.',file,row),
                facts_add(facts,'sector_jobs',int(sr.jobs),'postings',f'Total posting count in {sector}: {int(sr.jobs)}.',file,row),
                facts_add(facts,'sector_ai_share',float(sr.ai_rate),'share',f'AI-posting share in {sector}: {100*sr.ai_rate:.2f}%.',file,row)]
    # The derived ratio is stored in this immutable bundle, with raw-course and label hashes above.
    bundle.update(sector_ai_jobs=len(j),sector_all_jobs=int(sr.jobs),sector_ai_share=float(sr.ai_rate),sector_course_entries=len(sector_courses),
                  sector_courses_per_100_ai_jobs=ratio,course_sector_pattern=cpat,audience_title_matches=int(amask.sum()),market_fact_ids=market_ids,
                  total_ai_course_entries=len(courses),sector_course_ids=sector_courses.record_id.tolist(),
                  llm_reviewed_sector_jobs=int(j.reviewed_by_llm.sum()))
    js={k:j.search_text.str.contains(p,case=False,regex=True,na=False) for k,p in SKILLS.items()}
    cs={k:courses.review_text.str.contains(p,case=False,regex=True,na=False) for k,p in SKILLS.items()}
    for i,template in enumerate(CATALOGUE):
        skills=template['skills'];mask=pd.DataFrame({k:js[k] for k in skills}).any(axis=1)
        gaps=[max(0,float(js[k].mean())-float(cs[k].mean())) if len(j) else 0 for k in skills]
        overlap=sorted(set(template['levels'])&set(range(brief.level_min,brief.level_max+1)))
        depth=float(j.maturity.isin(overlap).mean()) if len(j) else 0
        compared=retrieve_courses(courses,sector+' '+template['name']+' '+brief.business_need,skills,limit=6)
        if brief.baseline_course_id and brief.baseline_course_id not in set(compared.record_id):
            b=courses[courses.record_id.eq(brief.baseline_course_id)].copy();b['similarity']=0.0
            compared=pd.concat([b,compared.head(5)],ignore_index=True)
        evidence=course_evidence(compared)
        known=sum(c['outcomes_available'] for c in evidence)
        parts={'brief_fit':min(1,float(fit[i])*3),'skill_gap':min(1,float(np.mean(gaps))*4),
               'skill_demand':float(mask.mean()) if len(j) else 0,'depth_fit':depth,'outcome_evidence':known/max(1,len(evidence)),
               'sector_opportunity':float(sr.Opportunity)}
        candidate={'candidate_id':'option_'+template['key'],'concept':template['name']+' · '+sector,'key':template['key'],
                   'skills':skills,'levels':template['levels'],'intent':template['intent'],'overlap_levels':overlap,
                   'sector_ai_jobs':len(j),'adaptation_route':brief.strategy=='Adapt an existing course' and bool(brief.baseline_course_id),'candidate_skill_jobs':int(mask.sum()),'explicit_support_share':support,
                   'sector_course_entries':len(sector_courses),'sector_courses_per_100_ai_jobs':ratio,
                   'audience_terms':terms,'audience_title_matches':int(amask.sum()),'sector_depth_share':depth,
                   'mean_positive_skill_gap':float(np.mean(gaps)),'brief_similarity':float(fit[i]),'known_outcome_courses':known,'baseline_outcomes_ok':not brief.baseline_course_id or any(c['course_id']==brief.baseline_course_id and c['outcomes_available'] for c in evidence),
                   'components':parts,'score':sum(POLICY['weights'][k]*v for k,v in parts.items()),'courses':evidence,
                   'skill_evidence':[{'skill':k,'sector_job_mentions':int(js[k].sum()),'sector_job_denominator':len(j),
                                      'pooled_course_mentions':int(cs[k].sum()),'pooled_course_denominator':len(courses),
                                      'outcome_courses_naming_skill':sum(any(re.search(SKILLS[k],o['quote'],re.I) for o in c['outcomes']) for c in evidence),
                                      'known_outcome_courses':known} for k in skills],
                   'job_examples':[{'job_id':r.record_id,'title':r.Title,'source_label':r.screening_source,
                                    'quote':str(r.Description)[:1000],'url':r['Job URL'] if isinstance(r['Job URL'],str) else None}
                                   for _,r in j[mask].head(3).iterrows()]}
        candidate['six_rules']=rule_results(candidate);candidate['eligible']=all(r['passed'] for r in candidate['six_rules'])
        bundle['candidates'].append(candidate)
    bundle['candidates'].sort(key=lambda c:c['score'],reverse=True)
    eligible=[c for c in bundle['candidates'] if c['eligible']]
    if eligible:bundle['recommended_candidate_id']=eligible[0]['candidate_id'];bundle['status']='available'
    bundle['sensitivity']=sensitivity(bundle['candidates'])
    bundle['mode_recommendation']='adapt' if brief.baseline_course_id else 'investigate adaptation before a new launch'
    return bundle


def analyse(brief):
    brief.validate()
    version=stamp()+((str(HERE/'policy.json'),(HERE/'policy.json').stat().st_mtime_ns,0),(str(HERE/'research.py'),(HERE/'research.py').stat().st_mtime_ns,0))
    # Return a detached value: UI changes cannot mutate the cached evidence.
    return json.loads(json.dumps(_analyse(json.dumps(brief.to_dict(),sort_keys=True),version),allow_nan=False))


def save_bundle(bundle):
    directory=DEST/'evidence';directory.mkdir(parents=True,exist_ok=True)
    identifier=digest(bundle);path=directory/(identifier+'.json')
    if not path.exists():path.write_text(json.dumps(bundle,indent=2,ensure_ascii=False,allow_nan=False))
    return identifier,path
