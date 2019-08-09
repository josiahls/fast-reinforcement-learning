from gym import envs


class Envs:

    @staticmethod
    def get_all_envs():
        return [env.id for env in envs.registry.all()]