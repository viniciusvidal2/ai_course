# ai_course

Repository for the AI course files.

## Base prompt

Use this prompt in your agent to generate the output content files:

``` bash
Follow the instructions given in @requests_prompts/base_instructions.md and @YOUR FILE to generate the output context file properly
```

## Organization of the repository

### requests_prompts

This folder contains the prompt files that will be used to generate the content files. Each prompt file corresponds to a specific topic or chapter of the course.

### content_documents

This folder contains the generated content files in Markdown format. Each file corresponds to a specific topic or chapter of the course, generated based on the prompts in the `requests_prompts` folder.

### images

This folder contains the images generated for the course content. Each image corresponds to a specific figure requested in the prompts and is referenced in the corresponding Markdown files in `content_documents`.
