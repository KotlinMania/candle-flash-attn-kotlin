pluginManagement {
    repositories {
        mavenLocal()
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

plugins { id("org.gradle.toolchains.foojay-resolver-convention") version "1.0.0" }

dependencyResolutionManagement {
    repositories {
        mavenLocal()
        google()
        mavenCentral()
    }
}

rootProject.name = "candle-flash-attn-kotlin"

listOf(
    "../candle-kotlin",
    "../candle-nn-kotlin",
).forEach { path ->
    if (file(path).exists()) {
        includeBuild(path)
    }
}
