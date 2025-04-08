"""Configurações do projeto. Não há necessidade de editar este arquivo a menos que você queira alterar
valores dos padrões do Kedro. Para mais informações, incluindo esses valores padrão, veja
https://docs.kedro.org/en/stable/kedro_project_setup/settings.html."""

# Hooks do projeto instanciados.
# from projetoinfnet.hooks import SparkHooks  # noqa: E402

# Hooks são executados em ordem Last-In-First-Out (LIFO).
HOOKS = () # SparkHooks removido

# Plugins instalados para os quais desabilitar o auto-registro de hooks.
# DISABLE_HOOKS_FOR_PLUGINS = ("kedro-viz",)

# Classe que gerencia o armazenamento de dados da KedroSession.
# from kedro.framework.session.store import BaseSessionStore
# SESSION_STORE_CLASS = BaseSessionStore
# Argumentos nomeados para passar ao construtor `SESSION_STORE_CLASS`.
# SESSION_STORE_ARGS = {
#     "path": "./sessions"
# }

# Diretório que contém a configuração.
# CONF_SOURCE = "conf"

# Classe que gerencia como a configuração é carregada.
from kedro.config import OmegaConfigLoader  # noqa: E402

CONFIG_LOADER_CLASS = OmegaConfigLoader
# Argumentos nomeados para passar ao construtor `CONFIG_LOADER_CLASS`.
CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
}
