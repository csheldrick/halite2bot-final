try:
    from .deathbot.bot import Bot
except:
    from bot import Bot
Bot(name="Deathbot").play()
