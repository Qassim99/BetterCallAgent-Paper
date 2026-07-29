"""Dependency construction for fixture and live online execution."""

from __future__ import annotations

from dataclasses import dataclass

from bettercallagent.citations.extract import CitationExtractor
from bettercallagent.citations.policy import FixedVotePolicy
from bettercallagent.citations.vocabulary import CitationVocabulary
from bettercallagent.providers.openai_compatible import (
    ChatProvider,
    OpenAICompatibleProvider,
)
from bettercallagent.settings import ExecutionMode, OnlineSettings
from online.fixture_provider import FixtureChatProvider
from online.repository import AssetError, OnlineAssetRepository


@dataclass(slots=True)
class OnlineDependencies:
    """Fully initialized dependencies shared by request-local pipeline runs."""

    settings: OnlineSettings
    repository: OnlineAssetRepository
    provider: ChatProvider
    extractor: CitationExtractor
    vocabulary: CitationVocabulary
    policy: FixedVotePolicy

    @classmethod
    def build(cls, settings: OnlineSettings) -> OnlineDependencies:
        """Build dependencies without any implicit mode fallback."""
        extractor = CitationExtractor()
        repository = OnlineAssetRepository.from_json(
            settings.asset_path,
            extractor=extractor,
        )
        if settings.mode is ExecutionMode.FIXTURE:
            if repository.fixture_script is None:
                raise AssetError("Fixture mode requires fixture_script in the configured asset.")
            provider: ChatProvider = FixtureChatProvider(repository.fixture_script)
        else:
            if settings.llm_base_url is None or settings.llm_api_key is None:
                raise RuntimeError("Validated live settings unexpectedly lack credentials.")
            provider = OpenAICompatibleProvider(
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,
                timeout_seconds=settings.request_timeout_seconds,
            )
        return cls(
            settings=settings,
            repository=repository,
            provider=provider,
            extractor=extractor,
            vocabulary=repository.vocabulary(),
            policy=FixedVotePolicy(),
        )

    async def close(self) -> None:
        """Release provider resources during application shutdown."""
        await self.provider.close()
