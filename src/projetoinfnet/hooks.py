# Este arquivo é o ponto de entrada principal para definir hooks customizados do Kedro.
# Para mais informações, por favor veja https://kedro.readthedocs.io/en/stable/hooks/introduction.html

# from kedro.framework.hooks import hook_impl
# from pyspark import SparkConf
# from pyspark.sql import SparkSession


# class SparkHooks:
#     @hook_impl
#     def after_context_created(self, context) -> None:
#         """Inicializa uma SparkSession usando a configuração
#         definida na pasta conf do projeto.
#         """

#         # Carrega a configuração spark em spark.yaml usando o config loader
#         parameters = context.config_loader["spark"]
#         spark_conf = SparkConf().setAll(parameters.items())

#         # Inicializa a sessão spark
#         spark_session_conf = (
#             SparkSession.builder.appName(context.project_path.name)
#             .enableHiveSupport()
#             .config(conf=spark_conf)
#         )
#         _spark_session = spark_session_conf.getOrCreate()
#         _spark_session.sparkContext.setLogLevel("WARN")
