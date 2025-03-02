# QAAI - Quality Assurance AI 🧪

> QAAI é uma ferramenta inovadora que utiliza IA para automatizar a geração de casos de teste de qualidade. Transforme descrições de funcionalidades em casos de teste estruturados com apenas alguns cliques, economizando tempo e mantendo a consistência na documentação de testes.

Uma ferramenta de geração de casos de teste baseada em inteligência artificial.

## Funcionalidades

- Geração de casos de teste baseados em descrições de funcionalidades
- Suporte a diferentes tipos de teste (Funcional, Unitário, Integração, E2E, Aceitação)
- Estruturação automática com pré-condições, passos e resultados esperados
- Exportação de casos de teste em formato JSON
- Interface amigável construída com Streamlit

## Tecnologias Utilizadas

- Python
- Streamlit
- LangChain
- OpenAI GPT-4
- Pydantic

## Como Executar

1. Clone o repositório
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure sua chave API da OpenAI no arquivo .env:
   ```
   OPENAI_API_KEY=sua_chave_api_aqui
   ```
4. Execute a aplicação:
   ```bash
   streamlit run projeto.py
   ```

## Estrutura dos Casos de Teste

Os casos de teste gerados incluem:
- Título
- Descrição detalhada
- Pré-condições necessárias
- Passos do teste
- Resultados esperados
- Tipo do teste
