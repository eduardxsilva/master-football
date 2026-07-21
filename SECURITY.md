# Segurança

## Credenciais

O projeto não deve armazenar chaves de API no código ou em arquivos versionados.
Configure `ODDS_API_KEY` e `API_FOOTBALL_KEY` como variáveis de ambiente ou
segredos privados da plataforma de hospedagem.

Se uma chave for publicada por engano:

1. revogue ou rotacione a chave imediatamente no provedor;
2. remova o segredo do código e do histórico do Git;
3. verifique os registros de uso da conta;
4. publique uma nova chave somente por um gerenciador de segredos.

## Relato de vulnerabilidade

Não publique chaves, dados pessoais ou detalhes exploráveis em uma issue pública.
Use o canal privado de contato disponibilizado pelo responsável do repositório.

## Dados locais

Planilhas importadas, relatórios exportados, páginas HTML, capturas de tela e
arquivos da pasta `debug_fifa_pages` podem conter dados coletados durante a
execução. Esses artefatos são ignorados pelo Git e devem ser revisados antes de
qualquer compartilhamento manual.
