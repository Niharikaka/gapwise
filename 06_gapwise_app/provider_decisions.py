"""Provider language over measured bundles; design ideas remain explicit hypotheses."""

DEFAULT_SCENARIO='expanded'

VIEW_LABELS={'Overview':'Your recommendation','Options':'Compare directions','Outcomes':'Existing courses','Proposal':'Course plan','Brief':'Change brief'}

DEPTH={1:'AI awareness',2:'guided use',3:'building and evaluating',4:'deployment',5:'production operations'}
DESIGNS={
 'documents':('Evidence-checked document answering','Answer workplace questions using approved documents, with citations and an escalation path for uncertain answers.','Test a small document-answering prototype against an approved question set. Submit the source passages, incorrect answers and proposed fixes.'),
 'assurance':('Practical AI governance and assurance','Review a workplace AI use case for privacy, fairness, accountability and approval requirements.','Produce a use-case risk review, control checklist and justified approval or escalation decision.'),
 'productivity':('Generative AI for everyday business work','Draft and improve workplace content, then check accuracy, privacy and suitability before use.','Submit a before-and-after workplace task with an output-checking log and corrections.'),
 'language':('AI for customer and text intelligence','Analyse workplace text and evaluate the quality and limitations of AI classifications or responses.','Evaluate a sample of text cases, document errors and define cases that need human review.'),
 'analytics':('AI-assisted analysis and forecasting','Use workplace data to explain patterns and compare a forecast with a simple baseline.','Submit a reproducible analysis, baseline comparison and a decision note explaining uncertainty.'),
 'workflow':('Building AI-assisted workflows','Connect retrieval and approved tools in a bounded workflow with human approval and recovery steps.','Demonstrate a workflow, its stop conditions, a failed-tool recovery and an audit trail.'),
 'prediction':('Predictive decision support','Compare predictive models with a baseline and decide whether errors are acceptable for the workplace task.','Produce an evaluation report with validation design, error analysis and operational constraints.'),
 'vision':('Computer vision for inspection','Evaluate image-based inspection for the target task and identify where human review is required.','Test inspection examples, analyse missed defects and false alarms, and propose a review threshold.'),
 'production':('Reliable AI deployment and monitoring','Operate an AI system with monitoring, incident handling and a rollback plan.','Demonstrate monitoring, a failure simulation and a documented recovery decision.'),
}

def selected(bundle):
    return next((c for c in bundle['candidates'] if c['candidate_id']==bundle.get('recommended_candidate_id') and c['eligible']),None)

def focus(bundle):
    return selected(bundle) or next(iter(bundle['candidates']),None)

def design(candidate):
    return DESIGNS[candidate['key']]

def rule_explanation(rule):
    o=rule['observed'];kind=rule['rule']
    if kind=='enough_demand':return f"{o['candidate_skill_jobs']:,} sector AI postings mention at least one focus skill; {o['explicit_support_share']:.0%} of sector AI postings have explicit AI evidence."
    if kind=='limited_supply':
        ratio=o['courses_per_100_ai_jobs']
        if ratio is None:return 'Sector-specific course supply is unknown. Obtain catalogue or syllabus evidence before judging market space.'
        return f"{o['sector_course_entries']:,} course titles map to this sector, versus {o['sector_ai_jobs']:,} AI postings ({ratio:.1f} per 100). "+('The brief is an adaptation; this does not establish a gap for a new launch.' if o['route']=='adapt existing' else 'This is an initial supply screen, not a measure of teaching capacity or competitor quality.')
    if kind=='clear_audience':return f"{o['title_matches']:,} sector AI job titles match learner keywords. Confirm the actual entry requirements with employers and learners."
    if kind=='right_depth':return f"{o['sector_depth_share']:.0%} of sector AI postings fall within the overlapping target depth."
    if kind=='skill_gap':return f"The mean positive job-minus-course mention difference is {o*100:.1f} percentage points. This compares sector jobs with the pooled AI course catalogue; it is not proof of missing teaching."
    return f"{o['known_outcome_courses']} retrieved courses have usable outcome passages. Brief similarity and source availability are screening checks, not evidence that this design is unique."

def coverage(candidate):
    rows=[]
    for r in candidate['skill_evidence']:
        known=r['known_outcome_courses'];mentions=r['outcome_courses_naming_skill']
        rows.append({'skill':r['skill'],'jobs':r['sector_job_mentions'],'job_total':r['sector_job_denominator'],
                     'named':mentions,'known':known,'missing':len(candidate['courses'])-known,
                     'meaning':('Build on existing teaching; test a workplace application and assessment.' if mentions else
                                'No match in the available outcome passages. Check full syllabi before treating this as an addition.') if known else 'Teaching coverage is unknown. Request the syllabus before proposing a gap.'})
    return rows

def next_actions(bundle):
    c=selected(bundle)
    if not c:
        failed=[]
        for option in bundle['candidates']:
            failed.extend(r['label'] for r in option['six_rules'] if not r['passed'])
        common=list(dict.fromkeys(failed))
        return [('Resolve the evidence blockers',', '.join(common[:3])+'.' if common else 'Choose a sector with analysed evidence.'),
                ('Check the audience and existing teaching','Confirm learner tasks and request full course syllabi. Change the brief only if the business need changes.'),
                ('Reassess before drafting','Repeat the local comparison after the missing evidence or brief has been resolved.')]
    known=c['known_outcome_courses'];total=len(c['courses'])
    return [('Check the existing teaching',f'Read the {known} usable course-outcome records in the {total}-course comparison set. Confirm the full syllabus and assessment coverage.'),
            ('Test the proposed workplace difference',design(c)[2]),
            ('Review the course plan',f'Draft the {bundle["brief"]["hours"]}-hour outline, then ask a provider reviewer to check learner fit, assessment, delivery resources and commercial assumptions.')]

def decision_markdown(bundle):
    c=selected(bundle);b=bundle['brief'];lines=['# Provider course decision',f"Sector: {bundle.get('sector') or b['sector']}",f"Learners: {b['audience']}",f"Workplace need: {b['business_need']}",f"Evidence scenario: {b['scenario']} · analysis version: {bundle['version']}",'',
        '## Suggested direction',design(c)[0] if c else 'Do not select a course yet: the initial evidence checks are not satisfied.',
        'This is a local analytical suggestion, not an AI-generated syllabus or an approved launch decision.','',
        '## What the current results say',f"Sector AI postings: {bundle.get('sector_ai_jobs','Unknown')} of {bundle.get('sector_all_jobs','Unknown')} sector postings.",
        f"Sector-mapped AI course entries: {bundle.get('sector_course_entries','Unknown')}. Mapping uses title terms and can miss relevant courses."]
    if c:
        lines += [f"Postings mentioning at least one focus skill: {c['candidate_skill_jobs']}.",'','## Existing teaching']
        for r in coverage(c):lines.append(f"- {r['skill']}: {r['named']} of {r['known']} compared courses with known outcome text name the skill. {r['meaning']}")
        lines += ['','## Possible assessed task — design hypothesis',design(c)[2]]
    lines += ['','## Next steps']+[f'{i}. {title}: {body}' for i,(title,body) in enumerate(next_actions(bundle),1)]
    return '\n\n'.join(lines)
