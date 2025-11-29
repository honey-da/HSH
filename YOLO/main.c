#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <winsock2.h>

#pragma comment(lib, "ws2_32.lib")

#define PORT 9000

int main() {
    WSADATA wsa;
    SOCKET server_fd, client_fd;
    struct sockaddr_in server_addr, client_addr;
    int client_len = sizeof(client_addr);
    char buffer[1024];

    // Winsock 초기화
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        printf("WSAStartup 실패\n");
        return 1;
    }

    // 소켓 생성
    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == INVALID_SOCKET) {
        printf("socket() 실패\n");
        WSACleanup();
        return 1;
    }

    // 주소 설정
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(PORT);

    // 바인드
    if (bind(server_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) == SOCKET_ERROR) {
        printf("bind() 실패\n");
        closesocket(server_fd);
        WSACleanup();
        return 1;
    }

    // 리슨
    if (listen(server_fd, 1) == SOCKET_ERROR) {
        printf("listen() 실패\n");
        closesocket(server_fd);
        WSACleanup();
        return 1;
    }

    printf("서버 실행 중... detect.py 연결 대기...\n");

    // 클라이언트 연결
    client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
    if (client_fd == INVALID_SOCKET) {
        printf("accept() 실패\n");
        closesocket(server_fd);
        WSACleanup();
        return 1;
    }

    printf("detect.py 연결됨!\n");

    // 메시지 반복 수신
    while (1) {
        memset(buffer, 0, sizeof(buffer));
        int n = recv(client_fd, buffer, sizeof(buffer), 0);

        if (n <= 0) {
            printf("클라이언트 종료됨\n");
            break;
        }

        printf("수신: %s\n", buffer);

        if (strstr(buffer, "WAKE")) {
            printf(">>> YOLO 감지됨! 알람 동작 실행!\n");

            // --------------------
            // 여기서 알람 동작 코드를 넣으면 됨
            // (릴레이 제어/모터/서보/스피커 등)
            // --------------------
        }
    }

    closesocket(client_fd);
    closesocket(server_fd);
    WSACleanup();
    return 0;
}
