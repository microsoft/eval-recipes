from pydantic import BaseModel


class InputContext(BaseModel):
    source_id: str
    title: str
    content: str


class InputClaimVerificationEvaluator(BaseModel):
    text: str  # The full text to analyze and split into sentences
    user_question: str  # The user question to provide context for the extraction
    source_context: list[InputContext]


class OutputCitation(BaseModel):
    source_id: str
    cited_text: str


class OutputSentenceSplittingStep(BaseModel):
    sentence: str
    start_index: int
    end_index: int
    preceding_sentences: list[str]
    following_sentences: list[str]
    sentence_with_surrounding_context: str  # The sentence with surrounding context


class OutputClaimVerificationStep(BaseModel):
    proof: str
    citations: list[OutputCitation]
    verified_probability: float = 0
    open_domain_justification: str = ""
    open_domain_probability: float = 0
    is_open_domain: bool = False


class OutputClaimVerificationEvaluator(BaseModel):
    sentence: str
    start_index: int  # Refers to the start index of the original sentence in the text
    end_index: int  # Refers to the end index of the original sentence in the text
    claim: str
    proof: str = ""  # The justification for the citations
    citations: list[OutputCitation] = []  # No citations indicates the claim is not supported
    verified_probability: float = 0  # Confidence that the claim is verified (0-100)
    open_domain_justification: str = ""  # Justification if an unverified claim is considered open-domain
    open_domain_probability: float = 0  # Confidence that the claim is open-domain (0-100)
    is_open_domain: bool = False  # Decision if the claim is open-domain or not


class OutputClaimVerificationEvaluatorMetrics(BaseModel):
    total_claims: int
    closed_domain_supported: float  # Percentage of claims that are supported, of the ones that are closed-domain (assumes supported are closed-domain)
    ignore_metric_recommended: bool  # Indicates if the metric has a high chance of being irrelevant to the input
    num_closed_domain_supported: int  # Number of closed-domain claims that are supported by the source context
    num_open_domain_claims: int  # Number of claims that are considered open-domain
    total_claims_closed_domain: int  # Total number of claims that are considered closed-domain
