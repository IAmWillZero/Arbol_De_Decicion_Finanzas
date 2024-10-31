package org.example;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.SerializationHelper;

import java.util.Random;
import java.util.logging.Logger;

public class Main {
    private static final Logger logger = Logger.getLogger(Main.class.getName());

    public static void main(String[] args) {

        try {
            // Cargar los datos
            logger.info("Cargando el archivo de datos...");
            DataSource source = new DataSource("data/ingresos.arff");
            Instances data = source.getDataSet();

            // Establecer el índice de la clase (último atributo)
            if (data.classIndex() == -1)
                data.setClassIndex(data.numAttributes() - 1);
            logger.info("Índice de clase configurado en: " + data.classAttribute().name());

            // Crear el árbol de decisión
            logger.info("Creando el modelo de árbol de decisión (J48)...");
            J48 tree = new J48(); // J48 es la implementación de C4.5
            tree.buildClassifier(data);
            logger.info("Modelo de árbol de decisión creado exitosamente.");

            // Evaluación del modelo con validación cruzada (10-fold cross-validation)
            logger.info("Evaluando el modelo con validación cruzada...");
            Evaluation evaluation = new Evaluation(data);
            evaluation.crossValidateModel(tree, data, 10, new Random(1));
            logger.info("Evaluación completada. Resultados:");
            System.out.println("=== Evaluación del Modelo ===");
            System.out.println("Correctly Classified Instances: " + evaluation.pctCorrect() + "%");
            System.out.println("Incorrectly Classified Instances: " + evaluation.pctIncorrect() + "%");
            System.out.println("Kappa statistic: " + evaluation.kappa());
            System.out.println("Mean absolute error: " + evaluation.meanAbsoluteError());
            System.out.println("Root mean squared error: " + evaluation.rootMeanSquaredError());
            System.out.println("Relative absolute error: " + evaluation.relativeAbsoluteError() + "%");
            System.out.println("Root relative squared error: " + evaluation.rootRelativeSquaredError() + "%");
            System.out.println("Summary: " + evaluation.toSummaryString());

            // Guardar el modelo
            logger.info("Guardando el modelo en archivo...");
            SerializationHelper.write("modelo_arbol_decision.model", tree);
            logger.info("Modelo guardado como 'modelo_arbol_decision.model'.");

            // Mostrar el árbol de decisión
            System.out.println("\n=== Árbol de Decisión ===");
            System.out.println(tree.toString());  // Visualización en texto del árbol

            // Mostrar más detalles sobre las hojas (clases terminales) y reglas
            System.out.println("\n=== Detalles de las Hojas y Reglas ===");
            for (int i = 0; i < data.numInstances(); i++) {
                double pred = tree.classifyInstance(data.instance(i));
                String actual = data.classAttribute().value((int) data.instance(i).classValue());
                String predicted = data.classAttribute().value((int) pred);
                System.out.printf("Instancia %d - Actual: %s, Predicho: %s%n", i + 1, actual, predicted);
            }

        } catch (Exception e) {
            logger.severe("Ocurrió un error durante la construcción o evaluación del modelo: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
