import optuna
import optuna.visualization as vis
import optunahub

if __name__ == "__main__":
    module = optunahub.load_module(package="samplers/auto_sampler")
    study = optuna.create_study(
        direction="maximize",
        study_name="CDEAgent-exhaustive",
        load_if_exists=True,
        storage="sqlite:///cde_agent.db",
        sampler=module.AutoSampler(),
    )
    print("Best trial:")
    print(study.best_trial)

    importances = optuna.importance.get_param_importances(study)
    top_5 = list(sorted(importances.keys(), key=lambda param: importances[param]))[:4]
    fig = vis.plot_contour(study, params=top_5)
    fig.show()
