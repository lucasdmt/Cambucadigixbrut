#include <cstring>
#include <cmath>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <cassert>
#include <pthread.h>
#include <fstream>
#include "GPU/GPUSecp.h"
#include "CPU/SECP256k1.h"
#include "CPU/HashMerge.cpp"
#include <sys/resource.h>
#include <chrono>

#include <cmath> //pow


long getFileContent(std::string fileName, std::vector<std::string> &vecOfStrs) {
	long totalSizeBytes = 0;

	std::ifstream in(fileName.c_str());
	if (!in)
	{
		//std::cerr << "Can not open the File : " << fileName << std::endl;
		return 0;
	}
	std::string str;
	while (std::getline(in, str))
	{
		vecOfStrs.push_back(str);
		totalSizeBytes += str.size();
	}
	
	in.close();
	return totalSizeBytes;
}

void loadInputHash(uint64_t *inputHashBufferCPU) {
	std::cout << "Loading hash buffer from file: " << NAME_HASH_BUFFER << std::endl;

	FILE *fileSortedHash = fopen(NAME_HASH_BUFFER, "rb");
	if (fileSortedHash == NULL)
	{
		printf("Error: not able to open input file: %s\n", NAME_HASH_BUFFER);
		exit(1);
	}

	fseek(fileSortedHash, 0, SEEK_END);
	long hashBufferSizeBytes = ftell(fileSortedHash);
	long hashCount = hashBufferSizeBytes / SIZE_LONG;
	rewind(fileSortedHash);

	if (hashCount != COUNT_INPUT_HASH) {
		printf("ERROR - Constant COUNT_INPUT_HASH is %d, but the actual hashCount is %lu \n", COUNT_INPUT_HASH, hashCount);
		exit(-1);
	}

	size_t size = fread(inputHashBufferCPU, 1, hashBufferSizeBytes, fileSortedHash);
	fclose(fileSortedHash);

	std::cout << "loadInputHash " << NAME_HASH_BUFFER << " finished!" << std::endl;
	std::cout << "hashCount: " << hashCount << ", hashBufferSizeBytes: " << hashBufferSizeBytes << std::endl;
}

void loadGTable(uint8_t *gTableX, uint8_t *gTableY) {
	std::cout << "loadGTable started" << std::endl;

	Secp256K1 *secp = new Secp256K1();
	secp->Init();

	for (int i = 0; i < NUM_GTABLE_CHUNK; i++)
	{
		for (int j = 0; j < NUM_GTABLE_VALUE - 1; j++)
		{
			int element = (i * NUM_GTABLE_VALUE) + j;
			Point p = secp->GTable[element];
			for (int b = 0; b < 32; b++) {
				gTableX[(element * SIZE_GTABLE_POINT) + b] = p.x.GetByte64(b);
				gTableY[(element * SIZE_GTABLE_POINT) + b] = p.y.GetByte64(b);
			}
		}
	}

	std::cout << "loadGTable finished!" << std::endl;
}

void salvarPosicoes(const char *strCPU, int *posicoesCPU, int *totalPosicoesCPU) {
    *totalPosicoesCPU = 0;
    for (int i = 0; strCPU[i] != '\0'; i++) {
        if (strCPU[i] == 'x') {
            posicoesCPU[*totalPosicoesCPU] = i;
            (*totalPosicoesCPU)++;
        }
    }
    printf("%s    Total de posições 'x' encontradas:%d\n",strCPU ,*totalPosicoesCPU);
    printf("X nas posições no array:");
    for (int i = 0; i < *totalPosicoesCPU; i++) {
        printf("%d ", posicoesCPU[i]); 
    }
    printf("    X nas posições na chave:");
    for (int i = 0; i < *totalPosicoesCPU; i++) {
        printf("%d ", posicoesCPU[i]+1); 
    }
    printf("\n");
}




void startSecp256k1ModeBooks(uint8_t * gTableXCPU, uint8_t * gTableYCPU, uint64_t * inputHashBufferCPU) {

  //char strCPU[65] = "6123ae95438e22e11b4a116b4c0c3d514ecf6cfede99370cabebf4f282b4228f";
	char strCPU[65] = "6x2xae9x438e22ex1b4a1x6b4c0c3d5x4ecf6cfede9x3x0cabebf4f282b4228f";
	int posicoesCPU[65];
	int totalPosicoesCPUtemp;

	salvarPosicoes(strCPU, posicoesCPU, &totalPosicoesCPUtemp);

	GPUSecp *gpuSecp = new GPUSecp(
		gTableXCPU,
		gTableYCPU,
		inputHashBufferCPU,
		strCPU,              
    	posicoesCPU,         
    	totalPosicoesCPUtemp
	);
	long timeTotal = 0;
	long totalCount = (COUNT_CUDA_THREADS);

	long long possibilidades = pow(10, totalPosicoesCPUtemp)-1;
	printf("possibilidades %lld \n",possibilidades);

	int itercount = COUNT_CUDA_THREADS * THREAD_MULT;
	int maxIteration = possibilidades / itercount;
	printf("cada iteração resulta em %d tentativas, resultando em no maximo de %d iterações \n", itercount, maxIteration);

	const char* original_key = "6123ae95418e22e11b4a116b4c0c3d514ecf6cfede99370cabebf4f282b4228f";
	    // Configurações
    const int POSICOES_FIXAS[] = {0, 8, 9, 10}; // Índices base 0
    const int TOTAL_POSICOES_FIXAS = 4;
    const int TOTAL_SUBSTITUICOES = 8;
    const int TAMANHO_CHAVE = 64;

    // Identifica posições de dígitos substituíveis
    int digit_positions[TAMANHO_CHAVE];
    int total_digits = 0;
    
    for (int i = 0; i < TAMANHO_CHAVE && original_key[i] != '\0'; i++) {
        int is_digit = (original_key[i] >= '0' && original_key[i] <= '9');
        int is_fixed = 0;
        for (int j = 0; j < TOTAL_POSICOES_FIXAS; j++) {
            if (POSICOES_FIXAS[j] == i) {
                is_fixed = 1;
                break;
            }
        }
        if (is_digit && !is_fixed) {
            digit_positions[total_digits++] = i;
        }
    }

    // Calcula combinações
    uint64_t num_combinacoes = 1;
    int n = total_digits;
    int k = TOTAL_SUBSTITUICOES;
    if (k > n) {
        num_combinacoes = 0;
    } else {
        for (int i = 1; i <= k; ++i) {
            num_combinacoes = num_combinacoes * (n - k + i) / i;
        }
    }
    printf("Dígitos substituíveis: %d\n", total_digits);
    printf("Combinações possíveis: %lu\n", num_combinacoes);


    // Buffer para a chave modificada
    char modified[TAMANHO_CHAVE + 1];
    strncpy(modified, original_key, TAMANHO_CHAVE);
    modified[TAMANHO_CHAVE] = '\0';

    // Implementação iterativa das combinações
    int indices[TOTAL_SUBSTITUICOES];
    
    // Inicializa os índices
    for (int i = 0; i < TOTAL_SUBSTITUICOES; i++) {
        indices[i] = i;
    }

    while (indices[0] <= total_digits - TOTAL_SUBSTITUICOES) {
        // Prepara a chave modificada
        strncpy(modified, original_key, TAMANHO_CHAVE);
        
        // Aplica as substituições
        for (int i = 0; i < TOTAL_SUBSTITUICOES; i++) {
            modified[digit_positions[indices[i]]] = 'x';
        }
        
        //rodar codigo aqui 
         gpuSecp->updateStrCPU(modified);

	for (int iter = 0; iter < maxIteration+1; iter++) {
		const auto clockIter1 = std::chrono::system_clock::now();
		gpuSecp->doIterationSecp256k1Books(iter);
		const auto clockIter2 = std::chrono::system_clock::now();
		gpuSecp->doPrintOutput();

		long timeIter1 = std::chrono::duration_cast<std::chrono::milliseconds>(clockIter1.time_since_epoch()).count();
		long timeIter2 = std::chrono::duration_cast<std::chrono::milliseconds>(clockIter2.time_since_epoch()).count();
		long iterationDuration = (timeIter2 - timeIter1);
		timeTotal += iterationDuration;

		totalCount = (itercount * (iter+1));

		int restantes = maxIteration - iter;
		long etaMillis = iterationDuration * restantes;

		int etaSeconds = etaMillis / 1000;
		int etaMinutes = etaSeconds / 60;
		int etaRemSeconds = etaSeconds % 60;

		int progresso = (int)((iter + 1) * 100.0 / (maxIteration + 1));
		static int lastPrintedPercent = -1;


		printf("Iteração %d/%d, Tempo: %ld ms, Progresso: %d%%, ETA: %d min %d s\r",iter + 1, maxIteration + 1, iterationDuration, progresso, etaMinutes, etaRemSeconds);

	}    //até aqui
        

        // Encontra o próximo conjunto de índices
        int t = TOTAL_SUBSTITUICOES - 1;
        while (t >= 0 && indices[t] == total_digits - TOTAL_SUBSTITUICOES + t) {
            t--;
        }
        
        if (t < 0) {
            break; // Todas as combinações foram geradas
        }
        
        indices[t]++;
        for (int j = t + 1; j < TOTAL_SUBSTITUICOES; j++) {
            indices[j] = indices[j-1] + 1;
        }
    }
	printf("Finished in %ld milliseconds (%.2f seconds)\n", timeTotal, timeTotal / 1000.0);
	printf("Total Seed Count: %lld \n", possibilidades);
	printf("Seeds Per Second: %0.2lf Mkeys\n", possibilidades / (double)(timeTotal * 1000));
}


void increaseStackSizeCPU() {
	const rlim_t cpuStackSize = SIZE_CPU_STACK;
	struct rlimit rl;
	int result;

	printf("Increasing Stack Size to %lu \n", cpuStackSize);

	result = getrlimit(RLIMIT_STACK, &rl);
	if (result == 0)
	{
		if (rl.rlim_cur < cpuStackSize)
		{
			rl.rlim_cur = cpuStackSize;
			result = setrlimit(RLIMIT_STACK, &rl);
			if (result != 0)
			{
				fprintf(stderr, "setrlimit returned result = %d\n", result);
			}
		}
	}
}

int main(int argc, char **argv) {
	printf("iniciando a cambuca \n");

	increaseStackSizeCPU();

	mergeHashes(NAME_HASH_FOLDER, NAME_HASH_BUFFER);

	uint8_t* gTableXCPU = new uint8_t[COUNT_GTABLE_POINTS * SIZE_GTABLE_POINT];
	uint8_t* gTableYCPU = new uint8_t[COUNT_GTABLE_POINTS * SIZE_GTABLE_POINT];

	loadGTable(gTableXCPU, gTableYCPU);

	uint64_t* inputHashBufferCPU = new uint64_t[COUNT_INPUT_HASH];

	loadInputHash(inputHashBufferCPU);

	startSecp256k1ModeBooks(gTableXCPU, gTableYCPU, inputHashBufferCPU);
	
	free(gTableXCPU);
	free(gTableYCPU);
	free(inputHashBufferCPU);

	printf("	Complete \n");
	return 0;
}
