from pydantic import BaseModel, Field
from typing import List

from .messages import SystemMessage


class CriterionEvaluation(BaseModel):
    arguments_for: List[str] = Field(
        default_factory=list,
        description="Specific examples, quotes, or aspects of the answer that support the score and meet the criterion's requirements (strengths).",
    )
    arguments_against: List[str] = Field(
        default_factory=list,
        description="Specific examples, quotes, or aspects of the answer that detract from the score or fail to meet the criterion's requirements (weaknesses/areas for improvement).",
    )
    score: int = Field(
        ge=1, le=5, description="Numerical score from 1 to 5 for this criterion."
    )
    rating_interpretation: str = Field(
        description="Explanation of what this specific score (1-5) means for this criterion, based on the scoring guide."
    )
    justification: str = Field(
        description="Detailed justification for the assigned score, explaining why the answer received this rating for this criterion. Be specific."
    )


class EvaluationSummary(BaseModel):
    key_strengths: List[str] = Field(
        default_factory=list,
        description="List the 2-3 most significant strengths of the answer overall.",
    )
    key_weaknesses: List[str] = Field(
        default_factory=list,
        description="List the 2-3 most significant weaknesses or areas for improvement overall.",
    )
    summary_comment: str = Field(
        description="A concise summary statement (2-3 sentences) about the overall quality of the answer, highlighting its main achievements and shortcomings."
    )
    overall_score: float = Field(
        ge=1.0,
        le=5.0,
        description="An overall score for the answer, considering all criteria. This should be a holistic assessment, potentially an average or a weighted reflection of the criteria scores.",
    )


class EvaluationResult(BaseModel):
    criterion_factual_accuracy: CriterionEvaluation = Field(
        description="Detailed evaluations for each factual accuracy criterion."
    )
    criterion_relevance: CriterionEvaluation = Field(
        description="Detailed evaluations for each relevance criterion."
    )
    criterion_completeness: CriterionEvaluation = Field(
        description="Detailed evaluations for each completeness criterion."
    )
    criterion_clarity_coherence: CriterionEvaluation = Field(
        description="Detailed evaluations for each clarity and coherence criterion."
    )
    criterion_instruction_following: CriterionEvaluation = Field(
        description="Detailed evaluations for each instruction following criterion."
    )

    summary: EvaluationSummary = Field(
        description="The overall assessment of the answer."
    )


PROMPT = """
You are an expert AI Evaluation Judge, acting as a stern and meticulous critic. Your task is to rigorously evaluate an AI-generated answer based on a given goal or task description. Your critique should be harsh but fair, identifying every possible area for improvement.
You MUST provide your evaluation in a structured JSON format according to the schema provided.

GENERAL INSTRUCTIONS:
- Analyze the goal/task and AI-generated answer thoroughly
- For arguments_for: Identify specific, genuine strengths that represent positive attributes
- For arguments_against: Adopt a highly critical perspective to uncover any weaknesses, ambiguities, oversights, or improvements, no matter how subtle
- Provide evidence-based justifications referencing specific parts of the answer
- Score each criterion from 1 to 5 based on the scale below

SCORING SCALE FOR ALL CRITERIA:
1: Very Poor - Fundamentally fails to meet the criterion with critical issues
2: Poor - Inadequate performance with significant issues that reduce usefulness
3: Fair - Meets basic requirements but has notable shortcomings
4: Good - Performs well with only minor, non-significant issues
5: Excellent - Meets criterion perfectly or almost perfectly with no significant issues

EVALUATION CRITERIA:

1. FACTUAL ACCURACY
   Focus: Truthfulness and verifiability of information; absence of fabrications or misleading statements
   Scoring Guide:
   1: Mostly inaccurate with significant misleading information
   2: Multiple factual errors or notable inaccuracies impacting credibility
   3: Generally accurate with some noticeable errors or minor inaccuracies
   4: Mostly accurate with only very minor, non-misleading inaccuracies
   5: Completely factually accurate and verifiable

2. RELEVANCE
   Focus: Direct alignment with the stated goal/task; absence of unnecessary information
   Scoring Guide:
   1: Completely irrelevant or fundamentally misinterprets the goal
   2: Largely irrelevant with substantial off-topic content
   3: Moderately relevant but includes off-topic information or misses aspects of the goal
   4: Mostly relevant with minimal irrelevant content
   5: Perfectly aligned with all aspects of the goal

3. COMPLETENESS
   Focus: Comprehensive coverage of all aspects of the goal/task with sufficient detail
   Scoring Guide:
   1: Grossly incomplete, missing critical aspects or extremely superficial
   2: Significantly incomplete, addressing only a portion of the goal
   3: Partially complete, addressing main points but missing some details or nuances
   4: Mostly complete with only minor omissions
   5: Fully comprehensive, addressing all aspects with appropriate depth

4. CLARITY & COHERENCE
   Focus: Logical structure, clear language, smooth flow, and grammatical correctness
   Scoring Guide:
   1: Very unclear, illogical, or incoherent with severe grammatical issues
   2: Unclear in many parts with significant structural or grammatical problems
   3: Reasonably clear but with some awkward phrasing or minor structural issues
   4: Clear, coherent, and well-structured with minimal grammatical errors
   5: Exceptionally clear, precise, and perfectly structured

5. INSTRUCTION FOLLOWING
   Focus: Adherence to all explicit and implicit instructions in the goal/task
   Scoring Guide:
   1: Completely ignores or fundamentally misunderstands instructions
   2: Fails to follow several key instructions
   3: Follows some instructions but misses or imperfectly implements others
   4: Follows most instructions with only minor deviations
   5: Perfectly follows all explicit and reasonably implicit instructions

---

This is the full interaction of the user with the AI including the goal/task and the AI-generated answer:
<message_history>
{history}
</message_history>
"""


class Judge:
    def __init__(self, llm) -> None:
        self.llm = llm
        self.llm.context_window_dynamic = True

    def evaluate(self, history:str) -> EvaluationResult:
        evalResult, _ = self.llm.chat(
            message=(
                "Start by summarizing what the user actually asked for and what ressources were made available to the AI. "
                "Then providing a detailed evaluation for each criterion, including arguments for and against the score. "
                "Then, summarize your overall assessment of the answer, highlighting key strengths and weaknesses. "
                "Finally, provide an overall score for the answer based on all criteria in the requested format."
            ),
            history=[
                SystemMessage(PROMPT.format(history=history)).print(),
            ],
            format=EvaluationResult,
        )
        return evalResult
