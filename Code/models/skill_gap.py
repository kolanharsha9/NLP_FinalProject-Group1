#%%

################################################################################################################
#  Below is the skill gap analysis that is being used in the the app
###############################################################################################################

import json
import spacy

# spaCy model
import spacy.cli
spacy.cli.download("en_core_web_md")
nlp = spacy.load("en_core_web_md")

def analyze_skill_gap(resume_json, job_description_json):
    
    # Parseing the JSON data
    resume_data = resume_json
    job_description_data =job_description_json

    resume_skills = resume_data.get("skills", [])
    job_skills = job_description_data.get("required_skills", [])

    missing_skills = [skill for skill in job_skills if skill not in resume_skills]

    resume_experience = [exp["responsibilities"] for exp in resume_data.get("work_experience", [])]
    resume_experience = [item for sublist in resume_experience for item in sublist]  
    resume_education = resume_data.get("education", [])
    job_experience = job_description_data.get("responsibilities", [])
    job_education = job_description_data.get("qualifications", [])

    experience_gap = []
    for job_exp in job_experience:
        job_exp_doc = nlp(job_exp)
        if not any(job_exp_doc.similarity(nlp(exp)) > 0.8 for exp in resume_experience):
            experience_gap.append(job_exp)

    education_gap = []
    for job_edu in job_education:
        job_edu_doc = nlp(job_edu)
        if not any(job_edu_doc.similarity(nlp(edu["degree"])) > 0.8 for edu in resume_education):
            education_gap.append(job_edu)

    feedback = ""
    if missing_skills:
        feedback += f"The following skills are missing from your resume: {', '.join(missing_skills)}\n"
    else:
        feedback += "\nYour resume matches all the required skills for the job.\n"
    
    if experience_gap:
        feedback += f"\nThe following work experience is missing from your resume: {', '.join(experience_gap)}\n"
    else:
        feedback += "\nYour resume matches all the required work experiences for the job.\n"

    if education_gap:
        feedback += f"\nThe following educational qualifications are missing from your resume: {', '.join(education_gap)}\n"
    else:
        feedback += "\nYour resume matches all the required educational qualifications for the job.\n"

    return feedback
# resume_json = '''
# {
#     "skills": [
#         "Python",
#         "MachineLearning",
#         "UtilizingChatGPTandotherAItools",
#         "Blender3DTool",
#         "Modelling",
#         "Composting",
#         "3Ddesigningandmodelling",
#         "3DAnimation"
#     ],
#     "work_experience": [
#         "2 years testing experience"
#     ],
#     "projects": [
#         {
#             "name": "AUTOMATICPERSONALITYRECOGNITIONINVIDEOINTERVIEWSUSINGCNN",
#             "description": "Developed an end-to-end interviewing model to perform automatic personality recognition (APR) during interviews.\\nThrought the input of interview video, this model will do screening process based on 5 personality traits.(OCEAN model)\\nImplemented by using visual and audio subnetworks in this project.\\nThe dataset used is First Impression V2, it consist of 10000 video files along with annotation files.",
#             "technologies": []
#         },
#         {
#             "name": "INTELLIGENTWASTESEGREGATIONTECHNIQUEUSINGCNN",
#             "description": "Developed a waste segregation model that can classify the waste into 9 different classes.\\nUsed a Deep Learning algorithm (VGG-16).\\nUsed MSW dataset and added our own images from google images to the dataset.",
#             "technologies": []
#         },
#         {
#             "name": "VISION",
#             "description": "Developed an application with smart assistance for blind people.\\nUsed COCO dataset and ssdmobilenetv2model for detecting objects.\\nThis application will detect objects around the victim and gives audio output of the object detected.\\nDeveloped the application using android studio. ashik.shaik.ali@gmail.com +1(571)-413-4739",
#             "technologies": []
#         },
#         {
#             "name": "AUDIOTRANSMISSIONTHROUGHLASER",
#             "description": "Developed a device that transmits audio through LASER.\\nThis works under the principle of intensity modulation and demodulation.\\nSolar plate is used as demodulator.",
#             "technologies": []
#         }
#     ],
#     "education": [
#         {
#             "institution": "GeorgeMasonUniversity",
#             "degree": "MS",
#             "graduation_year": "2024-25"
#         },
#         {
#             "institution": "VardhamanCollegeofEngineering",
#             "degree": "B-Tech",
#             "graduation_year": "2019-23"
#         },
#         {
#             "institution": "SriNalandaJuniorCollege",
#             "degree": "XII",
#             "graduation_year": "2017-19"
#         },
#         {
#             "institution": "ReginaCarmeliConventHighSchool",
#             "degree": "X",
#             "graduation_year": "2005-17"
#         }
#     ]
# }
# '''

# job_description_json = '''
# {
#     "skills": [
#         "Python",
#         "MachineLearning",
#         "DeepLearning",
#         "DataAnalysis",
#         "TensorFlow",
#         "Keras",
#         "NLP"
#     ],
#     "work_experience": [
#         "2+ years of experience in Machine Learning",
#         "Experience with TensorFlow and Keras"
#     ],
#     "education": [
#         {
#             "institution": "Any accredited university",
#             "degree": "MS",
#             "graduation_year": "Any"
#         }
#     ]
# }
# '''

# feedback = analyze_skill_gap(resume_json, job_description_json)
# print(feedback)
# %%
