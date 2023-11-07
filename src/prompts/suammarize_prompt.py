
def summarize_dense_entity(article, num_of_iterations):
    article = article
    prompt = f"""Article: [[{str(article)}]]\n\n You will generate increasingly concise, entity-dense summaries of the above Article. Repeat the following 2 steps {num_of_iterations} times. 
    Step 1. Identify 1-3 informative Entities (semicolon delimited) from the Article which are missing from the previously generated summary. 
    Step 2. Write a new, denser summary of identical length which covers every entity and detail from the previous summary plus the Missing Entities. 

    A Missing Entity is:
    - Relevant: to the main story.
    - Specific: descriptive yet concise (5 words or fewer).
    - Novel: not in the previous summary.
    - Faithful: present in the Article.
    - Anywhere: located anywhere in the Article.

    Guidelines:
    - The first summary should be long (4-5 sentences, ~80 words) yet highly non-specific, containing little information beyond the entities marked as missing. Use overly verbose language and fillers (e.g., "this article discusses") to reach ~80 words.
    - Make every word count: rewrite the previous summary to improve flow and make space for additional entities.
    - Make space with fusion, compression, and removal of uninformative phrases like "the article discusses".
    - The summaries should become highly dense and concise yet self-contained, e.g., easily understood without the Article.
    - Missing entities can appear anywhere in the new summary.
    - Never drop entities from the previous summary. If space cannot be made, add fewer new entities.

    Remember, use the exact same number of words for each summary.

    Answer in JSON. The JSON should be a list (length {num_of_iterations}) of dictionaries whose keys are "Missing_Entities" and "Denser_Summary"."""
    return prompt


def build_system_prompt():
    prompt = f"""You are an expert blog post writer, known for taking diverse research points and crafting them into engaging, 
    cohesive listicle articles. Given the assortment of research data provided, create an in-depth blog outline to guide the writing process. 
    This outline should weave together the research points in a logical and reader-friendly manner. Write in markdown format."""

    return prompt

def built_gen_sys_prompt(research):
    prompt = f"""You are an expert blog post writer, known for taking diverse research points and crafting them into engaging, 
    cohesive articles. Use these research points as additional data to the data provided in the prompt: {str(research)}"""

    return prompt

def build_outline_prompt(research):
    prompt = f"""Using the diverse research data provided here: [{str(research)}], 
create an engaging and in-depth blog outline that captures the essence of each point while maintaining a cohesive narrative. 
Ensure all points are addressed and the content flows seamlessly. Present your outline in markdown format."""
    return prompt

def build_article_prompt(outline):
    prompt = f"""Given the provided outline here: [{str(outline)}] and the research data at hand, craft a detailed listicle article. 
    Ensure that the content is both engaging and informative. 
    Incorporate lists, tables, and other markdown elements to enhance readability and provide value to the readers. 
    Maintain a consistent tone throughout and strive for coherence in presenting the data points. Write in markdown format. 
    Do not include anything in the response aside from the markdown. 
    Do not include a title for this article.
    Do not include any sort of filler or replacement text"""

    return prompt

def style_and_build_html(article):
    prompt = f"""Using HTML and CSS, style the article here: [{str(article)}] to make it more visually appealing and engaging. Write in HTML format"""
    return prompt