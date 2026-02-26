# Interactions

Performs a detailed security analysis of the input and/or output of LLM interactions.

Endpoint: POST /detection/v1/interactions
Version: 1
Security: HiddenLayerUserAuth

## Header parameters:

  - `HL-Project-Id` (string)
    The ID or alias for the Project to govern the request processing.
    Example: "internal-search-chatbot"

  - `X-Correlation-Id` (string)
    An ID that will be included with associated logs and downstream HTTP requests.
    Example: "6f22d397-6ca2-4359-8074-3318ab471fdf"

  - `X-Tenant-Id` (string, required)
    The ID of the tenant that the request is for, used in service-to-service calls.
    Example: "21c42040-40ed-4f01-9cea-963fee5eab3f"

## Request fields (application/json):

  - `metadata` (object, required)

  - `metadata.model` (string, required)
    The language model for the interactions.

  - `metadata.requester_id` (string, required)
    The identifier for the entity making the interactions.

  - `metadata.provider` (string)
    The provider of the language model.

  - `input` (object)

  - `input.messages` (array)
    The list of messages as input to a language model.

  - `input.messages.role` (string)
    The role of the message sender (e.g., user, assistant, system).

  - `input.messages.content` (string, required)
    The textual content of the message.

  - `output` (object)

  - `output.messages` (array)
    The list of messages as output from a language model.

## Response 200 fields (application/json):

  - `metadata` (object, required)

  - `metadata.event_id` (string)
    The unique identifier for the analysis event.

  - `metadata.analyzed_at` (string)
    The timestamp when the analysis was performed.

  - `metadata.provider` (string, required)
    The provider of the language model from the request.

  - `metadata.model` (string, required)
    The language model from the request.

  - `metadata.requester_id` (string, required)
    The identifier for the entity from the request.

  - `metadata.project` (object, required)

  - `metadata.project.project_id` (string)
    The unique identifier for the Project.

  - `metadata.project.project_alias` (string)
    A custom alias for the Project.

  - `metadata.project.ruleset_id` (string)
    The unique identifier for the Ruleset associated with the Project.

  - `metadata.processing_time_ms` (number, required)
    The total time taken to perform the analysis.

  - `analysis` (array, required)
    Example: [{"id":"prompt_injection.5.input","name":"prompt_injection","phase":"input","version":5,"configuration":{"enabled":true,"scan_type":"full","allow_overrides":{},"block_overrides":{}},"detected":true,"findings":{"frameworks":{"mitre":[{"label":"AML.T0051","name":"LLM Prompt Injection"}],"owasp":[{"label":"LLM01","name":"Prompt Injection"}],"owasp:2025":[{"label":"LLM01:2025","name":"Prompt Injection"}]},"probabilities":[1]},"processing_time_ms":7.01}]

  - `analysis.name` (string, required)
    The name of the analysis performed.
    Example: "prompt_injection"

  - `analysis.phase` (string, required)
    The phase of the analysis (i.e. input or output).
    Example: "input"

  - `analysis.version` (string, required)
    The version of the analysis performed.
    Example: 5

  - `analysis.detected` (boolean, required)
    Indicates the analysis resulted in a detection.
    Example: true

  - `analysis.configuration` (object, required)
    The configuration settings used for the analyzer.
    Example: {"enabled":true,"scan_type":"full","allow_overrides":{},"block_overrides":{}}

  - `analysis.findings` (object, required)
    The frameworks and associated findings for the analysis.
    Example: {"frameworks":{"mitre":[{"label":"AML.T0051","name":"LLM Prompt Injection"}],"owasp":[{"label":"LLM01","name":"Prompt Injection"}],"owasp:2025":[{"label":"LLM01:2025","name":"Prompt Injection"}]},"probabilities":[1]}

  - `analysis.findings.frameworks` (object, required)
    The taxonomies for the detections.
    Example: {"mitre":[{"label":"AML.T0051","name":"LLM Prompt Injection"}],"owasp":[{"label":"LLM01","name":"Prompt Injection"}],"owasp:2025":[{"label":"LLM01:2025","name":"Prompt Injection"}]}

  - `analysis.processing_time_ms` (number, required)
    The time taken to perform this specific analysis.
    Example: 7.01

  - `analysis.id` (string, required)
    The unique identifier for the analyzer.
    Example: "prompt_injection.5.input"

  - `evaluation` (object)
    The evaluation of the analysis results.

  - `evaluation.action` (string, required)
    The action based on interaction analysis and configured tenant security rules.
    Enum: "Allow", "Alert", "Redact", "Block"

  - `evaluation.has_detections` (boolean, required)
    Indicates if any detections were found during the analysis.

  - `evaluation.threat_level` (string, required)
    The threat level based on interaction analysis and configured tenant security rules.
    Enum: "None", "Low", "Medium", "High", "Critical"

  - `analyzed_data` (object, required)
    The language model input and/or output that was analyzed.

  - `analyzed_data.input` (object, required)

  - `analyzed_data.input.messages` (array)
    The list of messages as input to a language model.

  - `analyzed_data.input.messages.role` (string)
    The role of the message sender (e.g., user, assistant, system).

  - `analyzed_data.input.messages.content` (string, required)
    The textual content of the message.

  - `analyzed_data.output` (object)

  - `analyzed_data.output.messages` (array)
    The list of messages as output from a language model.

  - `modified_data` (object, required)
    The potentially modified language model input and output after applying any redactions or modifications based on the analysis.

## Response 422 fields (application/json):

  - `detail` (array)

  - `detail.loc` (array, required)

  - `detail.msg` (string, required)

  - `detail.type` (string, required)

