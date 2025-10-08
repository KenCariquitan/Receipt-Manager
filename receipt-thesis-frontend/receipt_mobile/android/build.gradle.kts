allprojects {
    repositories {
        google()
        mavenCentral()
    }
}
buildscript {
    val kotlin_version by extra("1.9.24") // Define extra property correctly
    repositories {
        google()
        mavenCentral()
    }
    dependencies {
        classpath("com.android.tools.build:gradle:8.4.2") // Use parentheses
        classpath("org.jetbrains.kotlin:kotlin-gradle-plugin:$kotlin_version") // Use parentheses and the variable
    }
}

val newBuildDir: Directory =
    rootProject.layout.buildDirectory
        .dir("../../build")
        .get()
rootProject.layout.buildDirectory.value(newBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    project.layout.buildDirectory.value(newSubprojectBuildDir)
}
subprojects {
    project.evaluationDependsOn(":app")
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}