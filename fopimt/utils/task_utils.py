from ..loader import Loader, Package
from ..message_repeating import MessageRepeatingTypeEnum
import json

"""
[ 
    UNDEFINED,0
    CONNECTOR_TYPE, 1
    EVALUATOR_TYPE, 2
    SOLUTION_TYPE, 3
    LANGUAGE_MODEL_TYPE, 4
    ANALYSIS_TYPE, 5
    STAT_TYPE, 6
    STOPPING_CONDITION_TYPE, 7
    TEST_TYPE, 8
    REPEATED_MESSAGE_TYPE, 9
    EVALUATOR_DYNAMIC_PARAM_TYPE 10
]
  """

def get_task_config(task_json: str, loader: Loader) -> str:

    task_dict = json.loads(task_json)

    # TODO Missing LANGUAGE_MODEL_TYPE, REPEATED_MESSAGE_TYPE ???
    db_types: dict[int, Package] = {1: loader.get_package_llmconnectors(),
                                    7: loader.get_package_stoppingconditions(),
                                    3: loader.get_package_solutions(),
                                    2: loader.get_package_evaluators(),
                                    5: loader.get_package_analysis(),
                                    8: loader.get_package_tests(),
                                    6: loader.get_package_stats(),
                                    4: loader.get_package_llmconnectors(),  # Special way to get LLM models
                                    9: 'REPEATED_MESSAGE_TYPE'  # Special way to get types of Rep. MSGs
                                    }

    # self._clear_all_options(options_id)

    # query = """
    # INSERT INTO "Data"."TaskConfigDynamicOptions" ("OptionsID", "Type", "ShortName", "LongName", "Description")
    # VALUES (%s, %s, %s, %s, %s)
    # """

    options = task_dict['config']['options']['configurationOptions']

    for key in db_types.keys():
        if key == 4:
            for modul in db_types[key].get_moduls():
                # Assuming that LLMConnector contains method 'get_all_models'
                _modul = db_types[key].get_modul_imported(modul.short_name)
                _models = getattr(_modul, 'get_all_models')()
                for _llmmodel in _models:

                    option = {
                        'type': key,
                        'shortName': _llmmodel.short_name,
                        'longName': _llmmodel.long_name,
                        'description': _llmmodel.description
                    }
                    options.append(option)

        elif key == 9:
            for member in MessageRepeatingTypeEnum:
                # TODO add long name and description to the Enum

                option = {
                    'type': key,
                    'shortName': member.name,
                    'longName': member.name,
                    'description': member.name
                }
                options.append(option)

        else:
            for modul in db_types[key].get_moduls():

                option = {
                    'type': key,
                    'shortName': modul.short_name,
                    'longName': modul.long_name,
                    'description': modul.description
                }
                options.append(option)

    task_json = json.dumps(task_dict)

    out = {
        loader.get_package_llmconnectors().get_name_class(): '',
        loader.get_package_solutions().get_name_class(): '',
        loader.get_package_analysis().get_name_class(): [],
        loader.get_package_evaluators().get_name_class(): '',
        loader.get_package_tests().get_name_class(): [],
        loader.get_package_stats().get_name_class(): [],
        loader.get_package_stoppingconditions().get_name_class(): [],
    }

    return task_json
