# This takes a student’s questionnaire responses, computes
# RIASEC scores according to the provided scoring key,
# then recommends the top 5 university majors and explains why
# they fit the student’s personality and preferences.

# Technical considerations:
# - The model is a locally hosted Zephyr-7B (Q5_0 quantized) running via llama.cpp for maximum inference efficiency.
# - The system uses LlamaCpp with n_ctx=4096 for a balance between context length and resource usage.
# - For cloud deployment (right now Google Cloud right now), resource optimization is critical:
#   * Set n_gpu_layers to >0 if GPU is available, or keep it 0 for CPU-only.
#   * Set n_batch appropriately (eg 64–128) to optimize token throughput vs. memory usage.
#   * Enable use_mmap and use_mlock for faster memory-mapped model loading.
# - FastAPI is used and I'm exposing a single POST endpoint, for the recommendation generation itself.
# - Dockerization is recommended, using a slim Python base image and minimal dependencies for smaller images and faster startup.
# - CPU throttling and minimum resource allocations are advised for cost-effective cloud deployment.
# - Logging token usage per request can be added later for monitoring and billing optimization if needed.

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
load_dotenv()

# Make sure to add the model path to your environment variables or set it here directly
MODEL_PATH = os.environ.get("MODEL_PATH", "models/zephyr-7b-beta.Q5_0.gguf")

# Few‑shot prompt template embedding:
#    - Instructions for mapping responses => RIASEC scores
#    - Mapping tables for the check‑box questions
#    - One complete worked example
#    - Final “Now you do it” section with placeholders

template = """
You are an expert career-counselor bot.  Your task:

1. **Map motivational responses** (6 Likert values) into RIASEC scores:
   - Intrinsic Motivation => Artistic (A), Investigative (I)
   - Identified Regulation => Social (S), Investigative (I), Artistic (A)
   - Integrated Regulation => Social (S), Enterprising (E), Artistic (A)
   - Introjected Regulation => Enterprising (E), Conventional (C), Realistic (R)
   - External Regulation => Conventional (C), Realistic (R), Enterprising (E)
   - Amotivation => (no categories)
   For each category, **add** the raw Likert value.

2. **Map check-box responses** (each +1 point) to RIASEC:
   - **Interest Fields**:
     • Health and medicine => S  
     • Agriculture and sciences => I  
     • Arts and communication => A  
     • Engineering and technology => R, I  
     • Business and management => E, C  
     • Human and public service => S  
   - **Qualities**:
     Compassionate and caring => S │ Good listener => S │ Following directions => C │ Conscientious => C │ Patient => S │
     Problem solver => I │ Nature lover => R │ Physically active => R │ Observer => I │ Imaginative => A │ Creative => A │
     Outgoing => S │ Performer => A │ Hands-on creator => R │ Logical thinker => I │ Practical => R │ Decision-maker => E │
     Open-minded => A │ Organized => C  
   - **Free-time Activities**:
     Volunteering => S │ Caring for others => S │ Healthy living => R │ Hiking => R │ Experimentation => I │
     Acting => A │ Writing => A │ Painting => A │ Building things => R │ Computing => I │ Coaching/tutoring => S

3. **Sum** all points for each of R, I, A, S, E, C.

4. **Choose top 2 RIASEC letters**. From each, pick majors:
   R => Mechanical Eng, Civil Eng, Electrical Eng, Architecture, Industrial Design  
   I => Biology, Chemistry, Computer Science, Mathematics, Data Science  
   A => Fine Arts, Graphic Design, Journalism, Music, Theater  
   S => Psychology, Nursing, Education, Social Work, Human Services  
   E => Business Admin, Marketing, Finance, Entrepreneurship, Management  
   C => Accounting, Finance, Economics, Library Science, Info Systems  

   Take **3 majors** from the highest-scoring letter and **2** from the second.

5. **Output** as a single paragraph:

---  
### Worked Example

**Student Responses**  
Interest Fields: Arts and communication, Human and public service  
Qualities: Compassionate and caring, Creative, Outgoing  
Free-time Activities: Writing, Acting, Volunteering  
Motivational Responses:  
- Intrinsic Motivation: 5  
- Identified Regulation: 4  
- Introjected Regulation: 2  
- Integrated Regulation: 3  
- Amotivation: 1  
- External Regulation: 2  

**Computation**  
- From Likert:  
A += 5 (Intrinsic) + 4 (Identified) + 3 (Integrated) = 12  
I += 5 + 4 = 9  
S += 4 + 3 + 2 = 9  
E += 2 + 2 = 4  
C += 2 + 2 = 4  
R += 2 = 2  
- From check-boxes:  
A += 1 (Arts) + 1 (Creative) + 1 (Outgoing) + 1 (Writing) + 1 (Acting) = 5 => A total = 17  
S += 1 (Human service) + 1 (Compassionate) + 1 (Outgoing) + 1 (Volunteering) = 4 => S total = 13  
… and so on.  

Final scores (example): A=17, S=13, I=9, E=4, C=4, R=2  
Top two letters: **A**, **S**  
Select 3 Artistic majors + 2 Social majors =>  
Top 5 majors: Fine Arts, Graphic Design, Music, Psychology, Nursing.

**Output Paragraph**:  
Top 5 majors: Fine Arts, Graphic Design, Music, Psychology, Nursing.  
This student shows exceptionally strong **Artistic** tendencies. Their creative qualities, like being imaginative and enjoying activities such as acting, writing, and painting, align perfectly with careers in fine arts and design. Their **Social** tendencies, showcased by their compassion and enjoyment of volunteering and helping others, make them well-suited for fields like psychology and nursing, where empathy and communication are key. This combination of creativity and social awareness suggests a fulfilling career path in the arts and human services.

---  
Now, process the actual student responses below and generate the recommendation.

**Student Responses**  
Interest Fields: {interest_fields}  
Qualities: {qualities}  
Free-time Activities: {free_time_activities}  
Motivational Responses:  
- Intrinsic Motivation: {intrinsic_motivation}  
- Identified Regulation: {identified_regulation}  
- Introjected Regulation: {introjected_regulation}  
- Integrated Regulation: {integrated_regulation}  
- Amotivation: {amotivation}  
- External Regulation: {external_regulation}  

**Return**: one paragraph with “Top 5 majors: …” then a long, insightful explanation.
Make sure to not use asterisks in your reply
Make sure to personalize it to them (use the pronoun "you", rather than referring to them as "This student")
"""

# Function for adding the variables (student responses) to the prompt template
prompt = PromptTemplate(
    input_variables=[
        "interest_fields",
        "qualities",
        "free_time_activities",
        "intrinsic_motivation",
        "identified_regulation",
        "introjected_regulation",
        "integrated_regulation",
        "amotivation",
        "external_regulation"
    ],
    template=template
)

# Initialize the LLM and chain. Temperature must be 0, so that output will be deterministic if the same information is given.
# Possible TODO: add n_gpu_layers and n_batch parameters to the LlamaCpp constructor for resource optimization
llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_ctx=4096,
    temperature=0.0
)
chain = LLMChain(llm=llm, prompt=prompt)

app = FastAPI()

# Request model based on how the answers will come, and in this order
class RIASECRequest(BaseModel):
    interest_fields: List[str]
    qualities: List[str]
    free_time_activities: List[str]
    intrinsic_motivation: int
    identified_regulation: int
    introjected_regulation: int
    integrated_regulation: int
    amotivation: int
    external_regulation: int

@app.post("/recommend")
def recommend(req: RIASECRequest):
    answers = req.model_dump()

    # Converting lists to strings for the prompt
    answers["interest_fields"] = ", ".join(answers["interest_fields"])
    answers["qualities"] = ", ".join(answers["qualities"])
    answers["free_time_activities"] = ", ".join(answers["free_time_activities"])

    out = chain.run(answers)
    return {"recommendation": out}
